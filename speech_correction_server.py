import os
import json
import logging
import asyncio
import hashlib
import hmac
import sqlite3
import time
from datetime import datetime, timedelta
from functools import lru_cache
from contextlib import asynccontextmanager, contextmanager
import threading
from dataclasses import dataclass

import regex as re
from fastapi import FastAPI, HTTPException, Request, Depends, Security
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, Dict, List
from langdetect import detect_langs, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

from openai import AsyncOpenAI, APIError, RateLimitError
from dotenv import load_dotenv
from cryptography.fernet import Fernet, InvalidToken
import firebase_admin
from firebase_admin import auth as firebase_auth
from firebase_admin import app_check as firebase_app_check
from firebase_admin import credentials as firebase_credentials
from language_catalog import CatalogError, display_name as catalog_display_name, load_catalog, runtime_views
from prompt_contract_v2 import AnalysisInput as V2AnalysisInput, AnalysisOutput as V2AnalysisOutput, CANONICAL_SYSTEM_PROMPT, build_user_payload

DetectorFactory.seed = 0
load_dotenv()

APP_VERSION = "2.2.0"

# Output budget for the model. Long texts with many errors produce large JSON
# payloads (Cyrillic explanations are token-expensive); 1500 used to truncate
# them mid-document, which surfaced to clients as "Error processing response".
DEEPSEEK_MAX_TOKENS = 4000
# Overridable via env so we can A/B a stronger model without a code change.
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
# Grammar correction is a focused task. DeepSeek V4 enables its thinking mode
# by default; disabling it avoids spending most of the request on hidden
# reasoning before it starts streaming the JSON response to the learner.
DEEPSEEK_THINKING = {"type": "disabled"}
# Bump either value whenever a change can affect the answer for the same input.
# Existing encrypted entries then become unreachable without manual cache work.
ANALYSIS_CONTRACT_VERSION = os.getenv("ANALYSIS_CONTRACT_VERSION", "v1").lower()
ANALYSIS_PROMPT_VERSION = os.getenv("ANALYSIS_PROMPT_VERSION", ANALYSIS_CONTRACT_VERSION)
ANALYSIS_SCHEMA_VERSION = "v2" if ANALYSIS_CONTRACT_VERSION == "v2" else "v1"

# Precompile dangerous input patterns once at module load
_DANGEROUS_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [
    r"<script\b[^>]*>", r"</script>", r"javascript:", r"eval\(", r"expression\(", r"on\w+\s*=",
    r"\{\{.*?\}\}", r"\$\{.*?\}", r"\$\(.*\)", r"`.*?`",
    r"\.\./", r"\.\.\\", r"%2e%2e", r"%252e",
    r"[‮‎‏‪‫‬‭]",
]]
# Allow common typographic punctuation across languages: straight AND curly
# apostrophes/quotes (iOS "smart punctuation" inserts curly ’ “ ” which used to
# be rejected as "invalid characters"), guillemets, en/em dashes, ellipsis,
# colon and semicolon. Dangerous characters (RTL overrides etc.) are still caught
# by _DANGEROUS_PATTERNS above.
_VALID_TEXT_RE = re.compile(
    r"""^[\p{L}\p{N}\s.,!?()'"’‘“”„‚«»–—…:;/-]+$""", re.UNICODE)

# Initialize DeepSeek async client
_deepseek_api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
_deepseek_client: Optional[AsyncOpenAI] = None
if _deepseek_api_key:
    _deepseek_client = AsyncOpenAI(
        api_key=_deepseek_api_key,
        base_url="https://api.deepseek.com/v1",
        timeout=30.0,
    )

# Authentication migration
#
# Released clients up to Android 1.6.5 use X-API-Key, which is not a secret once
# it has shipped in an APK/IPA. New clients send Firebase ID tokens instead. The
# legacy path remains only as a temporary compatibility bridge; set
# ALLOW_LEGACY_API_KEY=false after the migration window and rotate API_KEY.
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
_server_api_key = os.getenv("API_KEY")
_allow_legacy_api_key = os.getenv("ALLOW_LEGACY_API_KEY", "false").lower() == "true"
_firebase_project_id = os.getenv("FIREBASE_PROJECT_ID", "speechcorrection-4118e")
_firebase_service_account_json = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")
_firebase_ready = False
_app_check_enforced = os.getenv("APP_CHECK_ENFORCED", "true").lower() == "true"


def _initialize_firebase_admin() -> bool:
    """Initialise server-side Firebase token verification without a key file.

    Fly secrets hold the service-account JSON as one value. Application Default
    Credentials are also supported for local Google-managed environments.
    Failure is non-fatal during the migration: existing legacy clients keep
    working, but Bearer tokens are rejected until the secret is configured.
    """
    try:
        try:
            firebase_admin.get_app()
            return True
        except ValueError:
            pass
        if _firebase_service_account_json:
            certificate = json.loads(_firebase_service_account_json)
            firebase_admin.initialize_app(
                firebase_credentials.Certificate(certificate),
                {"projectId": _firebase_project_id},
            )
        else:
            firebase_admin.initialize_app(options={"projectId": _firebase_project_id})
        return True
    except Exception as error:
        logger.warning("Firebase Admin unavailable: %s", type(error).__name__)
        return False


@dataclass(frozen=True)
class AuthenticatedClient:
    """Trusted principal used for rate limits and audit-safe request metadata."""

    principal_id: str
    auth_scheme: str  # firebase | legacy
    app_check_status: str  # valid | missing | invalid | unavailable


def _bearer_token(request: Request) -> Optional[str]:
    value = request.headers.get("Authorization", "")
    scheme, _, token = value.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return None
    return token.strip()


def _verify_app_check(request: Request) -> str:
    """Verify an optional App Check token, enforcing only when enabled.

    App Check is intentionally deployed in monitoring mode first. The status is
    safe to log and exposes neither the token nor user-provided content.
    """
    token = request.headers.get("X-Firebase-AppCheck", "").strip()
    if not token:
        if _app_check_enforced:
            raise HTTPException(status_code=401, detail="Missing App Check token")
        return "missing"
    if not _firebase_ready:
        if _app_check_enforced:
            raise HTTPException(status_code=503, detail="App Check unavailable")
        return "unavailable"
    try:
        firebase_app_check.verify_token(token)
        return "valid"
    except Exception:
        if _app_check_enforced:
            raise HTTPException(status_code=401, detail="Invalid App Check token")
        return "invalid"


async def verify_client(request: Request, key: str = Security(_api_key_header)) -> AuthenticatedClient:
    """Accept a Firebase ID token, with a temporary legacy fallback.

    Never trust a UID supplied by the client: `verify_id_token` verifies the
    Firebase signature, audience, issuer and expiry before returning it.
    """
    app_check_status = _verify_app_check(request)
    request.state.app_check_status = app_check_status
    token = _bearer_token(request)
    if token:
        if not _firebase_ready:
            raise HTTPException(status_code=503, detail="Firebase authentication unavailable")
        try:
            decoded = firebase_auth.verify_id_token(token, check_revoked=False)
        except Exception:
            raise HTTPException(status_code=401, detail="Invalid or expired Firebase token")
        uid = decoded.get("uid")
        if not isinstance(uid, str) or not uid:
            raise HTTPException(status_code=401, detail="Firebase token has no user id")
        return AuthenticatedClient(
            principal_id=f"uid:{uid}",
            auth_scheme="firebase",
            app_check_status=app_check_status,
        )

    if _allow_legacy_api_key and _server_api_key and key == _server_api_key:
        return AuthenticatedClient(
            principal_id=f"legacy:{_get_client_ip(request)}",
            auth_scheme="legacy",
            app_check_status=app_check_status,
        )

    raise HTTPException(status_code=401, detail="Missing or invalid authentication")


# The versioned catalog is the only runtime language source.  The older JSON
# files remain in the repository temporarily as migration inputs/rollback data,
# but no endpoint reads them.
try:
    LANGUAGE_CATALOG = load_catalog()
except CatalogError as error:
    raise RuntimeError(f"Language catalog startup validation failed: {error}") from error
LANGUAGE_CONFIGS, INTERFACE_LANGUAGES, LEVEL_DETAILS, CONTEXT_INSTRUCTIONS = runtime_views(
    LANGUAGE_CATALOG)

# Configure logging. Never write user text or client IP addresses to a local
# file: platform log retention is configured outside the application.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_firebase_ready = _initialize_firebase_admin()

SERVER_URL = os.getenv("SERVER_URL", "https://speech-correction.fly.dev")
_is_prod = os.getenv("ENVIRONMENT", "production").lower() == "production"

app = FastAPI(
    title="Speech Correction API",
    description=f"Advanced API for language learning and speech correction. Base URL: {SERVER_URL}",
    version=APP_VERSION,
    docs_url=None if _is_prod else "/docs",
    redoc_url=None if _is_prod else "/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://speech-correction.fly.dev",
        "http://localhost:8080",
        "http://10.0.2.2:8080",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Grammar rule book endpoints (GET /rules, GET /rule) — served from static JSON
# in rules/ (see rules/README.md). Same Firebase/legacy migration auth as the
# other routes.
from rules_api import router as rules_router
import rules_store

app.include_router(rules_router, dependencies=[Depends(verify_client)])


class RateLimiter:
    def __init__(self, max_requests: int = 20, time_frame: timedelta = timedelta(minutes=1)):
        self.requests: Dict = {}
        self.max_requests = max_requests
        self.time_frame = time_frame
        self._lock = threading.Lock()

    def is_allowed(self, client_id: str) -> bool:
        with self._lock:
            now = datetime.now()
            self.requests = {k: v for k, v in self.requests.items() if now - v["first"] < self.time_frame}
            if client_id not in self.requests:
                self.requests[client_id] = {"count": 1, "first": now}
                return True
            record = self.requests[client_id]
            if record["count"] < self.max_requests:
                record["count"] += 1
                return True
            return False


class ResponseCache:
    """Bounded encrypted cache shared by identical requests.

    Values are encrypted at rest and database keys are HMACs, so neither a
    copied database nor a common-phrase dictionary reveals learner input. The
    cache has no user identifier: identical requests reuse paid model results.
    """

    _FORMAT_VERSION = "fernet-hmac-v1"

    def __init__(
        self,
        db_path: str = "response_cache.db",
        ttl_days: int = 30,
        max_entries: int | None = None,
        *,
        encryption_key: str | bytes | None = None,
        hmac_key: str | bytes | None = None,
    ):
        self.db_path = db_path
        self.ttl = timedelta(days=ttl_days)
        self.max_entries = max_entries if max_entries is not None else int(
            os.getenv("RESPONSE_CACHE_MAX_ENTRIES", "5000"))
        self._lock = threading.Lock()
        encryption_key = encryption_key or os.getenv("CACHE_ENCRYPTION_KEY")
        hmac_key = hmac_key or os.getenv("CACHE_HMAC_KEY")
        self._hmac_key = self._as_bytes(hmac_key)
        self._fernet: Fernet | None = None
        try:
            if encryption_key and self._hmac_key:
                self._fernet = Fernet(self._as_bytes(encryption_key))
        except (TypeError, ValueError):
            logger.error("Response cache disabled: invalid encryption configuration")
        self.enabled = self._fernet is not None and self._hmac_key is not None
        if not self.enabled:
            logger.warning(
                "Response cache disabled until CACHE_ENCRYPTION_KEY and CACHE_HMAC_KEY are configured")
            return
        if self.max_entries < 1:
            raise ValueError("RESPONSE_CACHE_MAX_ENTRIES must be positive")
        with self._lock, self._connection() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS cache ("
                "key TEXT PRIMARY KEY, response TEXT NOT NULL, created_at TEXT NOT NULL)"
            )
            conn.execute("CREATE TABLE IF NOT EXISTS cache_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            row = conn.execute("SELECT value FROM cache_meta WHERE key = 'format'").fetchone()
            if row is None or row[0] != self._FORMAT_VERSION:
                # Do not retain rows from the previous plaintext storage format.
                conn.execute("DELETE FROM cache")
                conn.execute(
                    "INSERT OR REPLACE INTO cache_meta (key, value) VALUES ('format', ?)",
                    (self._FORMAT_VERSION,),
                )

    @staticmethod
    def _as_bytes(value: str | bytes | None) -> bytes | None:
        if value is None:
            return None
        return value.encode("utf-8") if isinstance(value, str) else value

    def _digest(self, value: str) -> str:
        assert self._hmac_key is not None
        return hmac.new(self._hmac_key, value.encode("utf-8"), hashlib.sha256).hexdigest()

    @contextmanager
    def _connection(self):
        connection = sqlite3.connect(self.db_path)
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def make_key(self, text, language, level, style, interface_language, context) -> str:
        raw = json.dumps(
            [
                text, language, level, style, interface_language, context or "",
                DEEPSEEK_MODEL, ANALYSIS_PROMPT_VERSION, ANALYSIS_SCHEMA_VERSION,
            ],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return f"analysis::{self._digest(raw)}" if self.enabled else ""

    def _storage_key(self, key: str) -> str:
        return self._digest(key)

    def get(self, key: str) -> Optional[Dict]:
        if not self.enabled or not key:
            return None
        storage_key = self._storage_key(key)
        with self._lock, self._connection() as conn:
            row = conn.execute(
                "SELECT response, created_at FROM cache WHERE key = ?", (storage_key,)
            ).fetchone()
            if not row:
                return None
            response, created_at = row
            try:
                fresh = datetime.now() - datetime.fromisoformat(created_at) <= self.ttl
            except ValueError:
                fresh = False
            if not fresh:
                conn.execute("DELETE FROM cache WHERE key = ?", (storage_key,))
                return None
            try:
                assert self._fernet is not None
                return json.loads(self._fernet.decrypt(response.encode("utf-8")))
            except (InvalidToken, UnicodeDecodeError, json.JSONDecodeError):
                conn.execute("DELETE FROM cache WHERE key = ?", (storage_key,))
                logger.warning("Discarded unreadable encrypted cache entry")
                return None

    def put(self, key: str, response: Dict) -> None:
        if not self.enabled or not key:
            return
        try:
            assert self._fernet is not None
            encrypted = self._fernet.encrypt(
                json.dumps(response, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            ).decode("utf-8")
            with self._lock, self._connection() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO cache (key, response, created_at) VALUES (?, ?, ?)",
                    (self._storage_key(key), encrypted, datetime.now().isoformat()),
                )
                count = conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
                excess = count - self.max_entries
                if excess > 0:
                    conn.execute(
                        "DELETE FROM cache WHERE key IN (SELECT key FROM cache ORDER BY created_at ASC LIMIT ?)",
                        (excess,),
                    )
        except (sqlite3.Error, TypeError, ValueError) as error:
            # A cache write must never break a successful response.
            logger.error("Cache write failed: %s", type(error).__name__)


class LLMUsageMeter:
    """Privacy-safe, daily aggregate telemetry for paid model work.

    This deliberately records neither a Firebase UID nor text, prompt, response,
    IP address, or cache key. It only accounts for fixed product features.
    """

    _MICRODOLLARS = 1_000_000

    def __init__(
        self,
        db_path: str = "llm_usage.db",
        *,
        input_price_per_mtok: float | None = None,
        output_price_per_mtok: float | None = None,
    ):
        self.db_path = db_path
        self.input_price_per_mtok = (
            input_price_per_mtok
            if input_price_per_mtok is not None
            else float(os.getenv("LLM_INPUT_COST_PER_MTOK", "0"))
        )
        self.output_price_per_mtok = (
            output_price_per_mtok
            if output_price_per_mtok is not None
            else float(os.getenv("LLM_OUTPUT_COST_PER_MTOK", "0"))
        )
        if self.input_price_per_mtok < 0 or self.output_price_per_mtok < 0:
            raise ValueError("LLM token prices must be non-negative")
        self._lock = threading.Lock()
        with self._lock, self._connection() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS llm_usage_daily ("
                "day TEXT NOT NULL, feature TEXT NOT NULL, model TEXT NOT NULL, "
                "model_calls INTEGER NOT NULL DEFAULT 0, "
                "cache_hits INTEGER NOT NULL DEFAULT 0, "
                "prompt_tokens INTEGER NOT NULL DEFAULT 0, "
                "completion_tokens INTEGER NOT NULL DEFAULT 0, "
                "latency_ms INTEGER NOT NULL DEFAULT 0, "
                "estimated_cost_microdollars INTEGER NOT NULL DEFAULT 0, "
                "PRIMARY KEY(day, feature, model))"
            )

    @contextmanager
    def _connection(self):
        connection = sqlite3.connect(self.db_path)
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    @staticmethod
    def _safe_count(value: object) -> int:
        return value if isinstance(value, int) and value >= 0 else 0

    def _estimated_cost_microdollars(
        self, prompt_tokens: int, completion_tokens: int
    ) -> int:
        dollars = (
            prompt_tokens * self.input_price_per_mtok
            + completion_tokens * self.output_price_per_mtok
        ) / 1_000_000
        return round(dollars * self._MICRODOLLARS)

    def _upsert(
        self,
        *,
        feature: str,
        model: str,
        model_calls: int = 0,
        cache_hits: int = 0,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        latency_ms: int = 0,
    ) -> None:
        # feature and model are internal configuration values, but bounding them
        # prevents a bad configuration from growing the aggregate table forever.
        feature = str(feature)[:64]
        model = str(model)[:128]
        if not feature or not model:
            return
        estimated_cost = self._estimated_cost_microdollars(
            prompt_tokens, completion_tokens)
        try:
            with self._lock, self._connection() as conn:
                conn.execute(
                    "INSERT INTO llm_usage_daily ("
                    "day, feature, model, model_calls, cache_hits, prompt_tokens, "
                    "completion_tokens, latency_ms, estimated_cost_microdollars) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(day, feature, model) DO UPDATE SET "
                    "model_calls = model_calls + excluded.model_calls, "
                    "cache_hits = cache_hits + excluded.cache_hits, "
                    "prompt_tokens = prompt_tokens + excluded.prompt_tokens, "
                    "completion_tokens = completion_tokens + excluded.completion_tokens, "
                    "latency_ms = latency_ms + excluded.latency_ms, "
                    "estimated_cost_microdollars = "
                    "estimated_cost_microdollars + excluded.estimated_cost_microdollars",
                    (
                        datetime.now().date().isoformat(), feature, model,
                        model_calls, cache_hits, prompt_tokens, completion_tokens,
                        latency_ms, estimated_cost,
                    ),
                )
        except sqlite3.Error as error:
            # Accounting must never make a learner request fail.
            logger.error("LLM usage metric write failed: %s", type(error).__name__)

    def record_completion(
        self,
        *,
        feature: str,
        model: str,
        prompt_tokens: object,
        completion_tokens: object,
        latency_seconds: float,
    ) -> None:
        self._upsert(
            feature=feature,
            model=model,
            model_calls=1,
            prompt_tokens=self._safe_count(prompt_tokens),
            completion_tokens=self._safe_count(completion_tokens),
            latency_ms=max(0, round(latency_seconds * 1000)),
        )

    def record_cache_hit(self, *, feature: str, model: str) -> None:
        self._upsert(feature=feature, model=model, cache_hits=1)


def _get_client_ip(request: Request) -> str:
    # Fly.io sets Fly-Client-IP to the real client IP
    fly_ip = request.headers.get("Fly-Client-IP")
    if fly_ip:
        return fly_ip
    forwarded = request.headers.get("X-Forwarded-For", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host


# Register/style instructions appended to the prompt. Language-agnostic: the
# model honours these meta-instructions regardless of the target language, so
# the feature works for every language without touching the per-language
# prompt files.
STYLE_INSTRUCTIONS = {
    "formal": (
        "STYLE: Rewrite the text in a clean, standard formal register. Replace "
        "slang, colloquialisms and informal interjections with their neutral or "
        "literary equivalents while preserving the original meaning. The "
        "corrected_text must read as correct, formal language."
    ),
    "neutral": (
        "STYLE: Treat the text as everyday neutral speech. Fix clear grammatical "
        "errors and wrong word choices, but accept natural conversational phrasing "
        "— do not rewrite casual-but-correct sentences into bookish or overly "
        "formal language. Keep corrected_text in the same neutral register."
    ),
    "casual": (
        "STYLE: The user is intentionally speaking in a casual/informal register "
        "(slang, colloquialisms, contractions). PRESERVE that register: do NOT "
        "treat slang or colloquial expressions as errors and do NOT convert them "
        "to formal language. Only correct genuine mistakes that break grammar or "
        "meaning, and keep corrected_text in the same informal register the user "
        "used."
    ),
}

# Appended (language-agnostic) when the user provides a context/intent. Turns the
# context into the GOAL: produce the corrected_text from it, even from an empty
# input, so a stuck learner can write their meaning in their own language and get
# the right sentence back.
INTENT_INSTRUCTION = (
    "INTENT MODE: The context above states what the user is TRYING to say (their "
    "intended meaning, possibly written in their own language). Treat it as the "
    "goal. corrected_text MUST express that intended meaning naturally and "
    "correctly in the target language. If the user's text is empty, very short, "
    "or only partially expresses the intent, DO NOT answer that there is no text "
    "to analyze — compose corrected_text yourself from the intended meaning. "
    "error_analysis should capture the gap between what the user wrote and the "
    "intended meaning (missing words, wrong choices); if the user wrote nothing it "
    "may be empty. Make 'alternatives' natural ways to express the same intent, "
    "and use 'level_appropriate_suggestions' to point out words or phrases the "
    "user was missing."
)


# Appended to every prompt. Roughly 60% of the output (and therefore of the
# response latency, since output tokens are generated sequentially) is spent on
# the four teaching fields. Capping their length keeps the high-signal core while
# cutting generation time — a dense two-sentence note usually helps a learner
# more than a rambling paragraph. Tunable: relax the caps to trade speed for
# depth, or remove this block to restore the previous verbosity.
BREVITY_INSTRUCTION = (
    "RESPONSE LENGTH: Keep 'explanation', 'grammar_notes', 'pronunciation_tips' "
    "and 'level_appropriate_suggestions' concise — at most 2 sentences each, "
    "focused on what most helps the learner. Provide at most 2 'alternatives'. "
    "Be precise and useful, not verbose. This does NOT apply to 'error_analysis': "
    "still report EVERY error."
)

# Appended to every prompt. The learner's level was subtly steering the model to
# only fix "level-relevant" grammar and leave other real errors in corrected_text
# (e.g. at B2 it fixed the participle "fahren"->"gefahren" but kept the wrong
# auxiliary "habe" instead of "bin"). corrected_text must always be fully correct;
# the level only changes how we EXPLAIN, not how completely we fix.
CORRECTNESS_INSTRUCTION = (
    "CORRECTNESS OVERRIDES LEVEL: corrected_text MUST be fully correct and natural "
    "regardless of the learner's level — never leave ANY real error unfixed "
    "(grammar, auxiliary/verb choice, agreement, case, word order, vocabulary, "
    "spelling, punctuation). The level ONLY tailors the depth and focus of your "
    "explanations and level_appropriate_suggestions; it must NOT make you correct "
    "less thoroughly. Report every fix you make in error_analysis."
)

# Appended to every prompt. The client highlights each correction in place by
# finding its 'corrected' fragment inside corrected_text; if the model paraphrases
# instead of copying, nothing matches and the correction falls back to a plain
# list (this is why full rewrites, e.g. German, showed no inline highlights).
HIGHLIGHT_INSTRUCTION = (
    "ANCHORING (important for the UI): in every error_analysis item, 'corrected' "
    "MUST be copied VERBATIM from corrected_text (an exact contiguous substring — "
    "same words, spelling, case and punctuation), and 'original' an exact "
    "substring of the user's input. Pick the smallest span that captures the fix. "
    "Do not paraphrase these two fields."
)

# Appended only when the input came from speech recognition with low confidence.
# Without it the model treats transcription noise as learner mistakes; with it,
# obvious mis-hearings are read as the intended word instead of flagged. {conf}
# is the recognition_confidence the client already sends (previously ignored).
LOW_CONFIDENCE_INSTRUCTION = (
    "SPEECH INPUT NOTE: This text was transcribed from speech with LOW "
    "recognition confidence ({conf:.2f}). Some apparent errors may be "
    "transcription artifacts, not the learner's mistakes. When a 'mistake' looks "
    "like a likely mis-transcription of a correct word in context, assume the "
    "intended word and do NOT flag it as a learner error."
)
# Below this recognition_confidence we warn the model about transcription noise.
_LOW_CONFIDENCE_THRESHOLD = 0.7


class CorrectionRequest(BaseModel):
    text: str
    language: str
    level: str
    interface_language: str
    recognition_confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    context: Optional[str] = None
    # Target register. Defaults to "formal" (clean standard language) for
    # clients that don't send a style.
    style: str = "formal"

    @field_validator("interface_language")
    def validate_interface_language(cls, v):
        if v not in INTERFACE_LANGUAGES:
            raise ValueError(f"Unsupported interface language: {v}")
        return v

    @field_validator("language")
    def validate_language(cls, v):
        if v not in LANGUAGE_CONFIGS:
            raise ValueError(f"Unsupported language: {v}")
        return v

    @field_validator("level")
    def validate_level(cls, v):
        if v not in LEVEL_DETAILS:
            raise ValueError(f"Unsupported level: {v}")
        return v

    @field_validator("style")
    def validate_style(cls, v):
        if v not in STYLE_INSTRUCTIONS:
            raise ValueError(f"Unsupported style: {v}")
        return v

    @field_validator("text")
    def validate_text(cls, v):
        v = v.strip()
        if not v:
            # Empty text is allowed ONLY when an intent/context is provided
            # (see require_text_or_context). Skip the content checks below.
            return v
        if len(v) > 1000:
            raise ValueError(f"Text is too long ({len(v)} characters). Maximum allowed is 1000.")
        for pattern in _DANGEROUS_PATTERNS:
            if pattern.search(v):
                raise ValueError("Potentially dangerous constructs detected")
        if not _VALID_TEXT_RE.match(v):
            raise ValueError("Text contains invalid characters")
        return v

    @field_validator("context")
    def validate_context(cls, v):
        if v is None:
            return v
        v = v.strip()
        if len(v) > 2000:
            raise ValueError(f"Context is too long ({len(v)} characters). Maximum allowed is 2000.")
        for pattern in _DANGEROUS_PATTERNS:
            if pattern.search(v):
                raise ValueError("Potentially dangerous constructs detected in context")
        return v

    @model_validator(mode="after")
    def require_text_or_context(self):
        # Intent mode: the user may leave the text empty as long as they say what
        # they want to express (context). But both can't be empty.
        if not (self.text or "").strip() and not (self.context or "").strip():
            raise ValueError("Provide some text or describe what you want to say")
        return self


@lru_cache(maxsize=32)
def load_prompt_template(language: str) -> Dict:
    prompt_file = f"prompts/prompt_{language}.json"
    try:
        with open(prompt_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Prompt file for {language} not found")
        raise HTTPException(status_code=500, detail=f"Prompt file for {language} not found")
    except json.JSONDecodeError:
        logger.error(f"Invalid prompt file for {language}")
        raise HTTPException(status_code=500, detail=f"Invalid prompt file for {language}")


def generate_teacher_prompt(request: CorrectionRequest, retry: bool = False) -> str:
    prompt_data = load_prompt_template(request.language)
    prompt_template = prompt_data["prompt"]

    interface_lang_config = INTERFACE_LANGUAGES.get(request.interface_language, INTERFACE_LANGUAGES["en"])
    level_info = LEVEL_DETAILS[request.level]
    lang_config = LANGUAGE_CONFIGS[request.language]

    if request.context:
        context_instruction_template = CONTEXT_INSTRUCTIONS[request.language]["with_context"]
        context_instruction = context_instruction_template.format(
            context=request.context,
            interface_language=interface_lang_config["name"]
        )
    else:
        context_instruction = CONTEXT_INSTRUCTIONS[request.language]["no_context"]

    try:
        prompt = prompt_template.format(
            level=request.level,
            text=request.text,
            interface_language=interface_lang_config["name"],
            interface_language_code=interface_lang_config["language_code"],
            level_description=_level_description(level_info, interface_lang_config),
            common_errors=", ".join(lang_config["common_errors"]),
            pronunciation_focus=", ".join(lang_config["pronunciation_focus"]),
            grammar_focus=", ".join(level_info["grammar_focus"]),
            context_instruction=context_instruction
        )
    except KeyError as e:
        logger.error(f"Missing key in prompt formatting: {e}")
        raise HTTPException(status_code=500, detail=f"Error formatting prompt: missing key {e}")

    # Append the register/style instruction for the requested style.
    style_instruction = STYLE_INSTRUCTIONS.get(request.style, "")
    if style_instruction:
        prompt += f"\n\n{style_instruction}"

    # Intent mode: when the user said what they want to express, make the model
    # produce/repair the sentence toward that meaning.
    if request.context:
        prompt += f"\n\n{INTENT_INSTRUCTION}"

    # Keep the teaching fields tight (speed) without touching error coverage.
    prompt += f"\n\n{BREVITY_INSTRUCTION}"

    # The level must not reduce how completely we correct.
    prompt += f"\n\n{CORRECTNESS_INSTRUCTION}"

    # Make corrections anchorable so the client can highlight them in place.
    prompt += f"\n\n{HIGHLIGHT_INSTRUCTION}"

    # Speech transcribed with low confidence: don't mistake mis-hearings for the
    # learner's errors.
    if request.recognition_confidence < _LOW_CONFIDENCE_THRESHOLD:
        prompt += "\n\n" + LOW_CONFIDENCE_INSTRUCTION.format(
            conf=request.recognition_confidence)

    if retry:
        prompt += (
            f"\n\nWARNING: Previous response contained explanations in the wrong language. "
            f"ALL explanations, grammar_notes, pronunciation_tips, level_appropriate_suggestions, "
            f"and error_analysis explanations MUST be in {interface_lang_config['name']} "
            f"(ISO: {interface_lang_config['language_code']})."
        )

    return prompt


def _v2_mapping_threshold() -> float:
    """Configured confidence gate for showing a rule; invalid values fail safe."""
    try:
        value = float(os.getenv("RULE_MAPPING_MIN_CONFIDENCE", "0.75"))
    except ValueError:
        return 0.75
    return min(1.0, max(0.0, value))


def build_v2_prompt(request: CorrectionRequest) -> tuple[str, str]:
    """Build instruction-only system text plus separate structured learner data."""
    concepts = rules_store.topics_with_concepts(request.language)
    concept_catalog = [
        {"concept_code": topic["concept_code"], "title": topic["title"]}
        for topic in concepts
    ]
    system = (
        f"{CANONICAL_SYSTEM_PROMPT}\n\n"
        "Allowed concept codes for this learning language follow. Choose one only "
        "when the observed correction is directly taught by that topic; otherwise "
        "use taxonomy.unresolved.\n"
        f"{json.dumps(concept_catalog, ensure_ascii=False, separators=(',', ':'))}"
    )
    payload = V2AnalysisInput(
        text=request.text,
        learning_language=request.language,
        interface_language=request.interface_language,
        level=request.level,
        style=request.style,
        context=request.context,
    )
    return system, build_user_payload(payload)


def parse_v2_correction_response(response: str, request: CorrectionRequest) -> Dict:
    """Validate V2 output and adapt it to the response shape used by released apps."""
    parsed = V2AnalysisOutput.model_validate(_extract_json_object(response))
    errors: list[dict] = []
    threshold = _v2_mapping_threshold()
    for error in parsed.errors:
        # Anchors are a product invariant: they make text highlighting accurate
        # and prevent a model rewrite from being presented as a small correction.
        if request.text and error.original not in request.text:
            raise ValueError("V2 original anchor is not in submitted text")
        if error.corrected not in parsed.corrected_text:
            raise ValueError("V2 corrected anchor is not in corrected text")
        rule_id = (
            rules_store.resolve_concept(request.language, error.concept_code)
            if error.confidence >= threshold
            else None
        )
        errors.append({
            "type": error.category.value,
            "concept_code": error.concept_code,
            "confidence": str(error.confidence),
            "rule_id": rule_id or "",
            "original": error.original,
            "corrected": error.corrected,
            "explanation": error.explanation,
        })
    counts: dict[str, int] = {}
    for error in errors:
        counts[error["type"]] = counts.get(error["type"], 0) + 1
    return {
        "contract_version": 2,
        "corrected_text": parsed.corrected_text,
        "error_analysis": errors,
        "error_statistics": ", ".join(
            f"{category}: {count}" for category, count in sorted(counts.items())),
        "explanation": parsed.summary,
        "grammar_notes": "",
        "pronunciation_tips": "",
        "alternatives": "",
        "level_appropriate_suggestions": "",
    }


def _level_description(level_info: Dict, interface_config: Dict) -> str:
    """Read legacy display-name descriptions through a code-first adapter.

    Catalog v2 will use ISO codes directly. Until then this keeps every current
    interface localized without language-specific branches.
    """
    descriptions = level_info.get("description", {})
    if not isinstance(descriptions, dict):
        return ""
    return (
        descriptions.get(interface_config.get("language_code"))
        or descriptions.get(interface_config.get("name"))
        or descriptions.get("English", "")
    )


_CODE_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$")


def _extract_json_object(response: str) -> Dict:
    """Parse the model output as a JSON object, tolerating markdown fences
    and surrounding prose. Raises ValueError if no valid object is found."""
    candidates = [response, _CODE_FENCE_RE.sub("", response)]
    start, end = response.find("{"), response.rfind("}")
    if start != -1 and end > start:
        candidates.append(response[start:end + 1])
    for candidate in candidates:
        try:
            result = json.loads(candidate)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            continue
    raise ValueError("Model response is not a valid JSON object")


# Matches a COMPLETE JSON string value for corrected_text in a partial buffer
# (corrected_text is the first field the model emits). Used to surface the
# correction to the client the moment it finishes, before the slower teaching
# fields are generated.
_CORRECTED_TEXT_RE = re.compile(r'"corrected_text"\s*:\s*"((?:[^"\\]|\\.)*)"')


async def _call_deepseek_stream(
    client: AsyncOpenAI, prompt: str, user_text: str, *, feature: str = "analysis_stream"
):
    """Yield the model's content deltas as they arrive (same params as the
    non-streaming call, so output is identical — just incremental)."""
    started = time.perf_counter()
    first_content_seconds: Optional[float] = None
    content_chars = 0
    usage = None
    stream = await client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_text},
        ],
        temperature=0.3,
        max_tokens=DEEPSEEK_MAX_TOKENS,
        response_format={"type": "json_object"},
        stream=True,
        stream_options={"include_usage": True},
        extra_body={"thinking": DEEPSEEK_THINKING},
    )
    try:
        async for chunk in stream:
            if getattr(chunk, "usage", None):
                usage = chunk.usage
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                if first_content_seconds is None:
                    first_content_seconds = time.perf_counter() - started
                    logger.info(
                        "DeepSeek stream first_content_seconds=%.3f model=%s",
                        first_content_seconds,
                        DEEPSEEK_MODEL,
                    )
                content_chars += len(delta)
                yield delta
    finally:
        elapsed = time.perf_counter() - started
        llm_usage_meter.record_completion(
            feature=feature,
            model=DEEPSEEK_MODEL,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            latency_seconds=elapsed,
        )
        logger.info(
            "DeepSeek stream completed model=%s first_content_seconds=%s "
            "total_seconds=%.3f content_chars=%s prompt_tokens=%s completion_tokens=%s",
            DEEPSEEK_MODEL,
            f"{first_content_seconds:.3f}" if first_content_seconds is not None else "none",
            elapsed,
            content_chars,
            getattr(usage, "prompt_tokens", None),
            getattr(usage, "completion_tokens", None),
        )


async def _call_deepseek(client: AsyncOpenAI, prompt: str, user_text: str,
                         *, feature: str = "generic",
                         max_tokens: int = DEEPSEEK_MAX_TOKENS) -> str:
    started = time.perf_counter()
    response = await client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_text},
        ],
        # Correction is a precision task, not a creative one. A low temperature
        # makes the model correct rather than rewrite, and produces more
        # consistent, reliably-parseable JSON (fewer parse-retry round-trips).
        temperature=0.3,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
        extra_body={"thinking": DEEPSEEK_THINKING},
    )
    choice = response.choices[0]
    usage = getattr(response, "usage", None)
    elapsed = time.perf_counter() - started
    llm_usage_meter.record_completion(
        feature=feature,
        model=DEEPSEEK_MODEL,
        prompt_tokens=getattr(usage, "prompt_tokens", None),
        completion_tokens=getattr(usage, "completion_tokens", None),
        latency_seconds=elapsed,
    )
    logger.info(
        "DeepSeek completion feature=%s model=%s total_seconds=%.3f "
        "prompt_tokens=%s completion_tokens=%s",
        feature,
        DEEPSEEK_MODEL,
        elapsed,
        getattr(usage, "prompt_tokens", None),
        getattr(usage, "completion_tokens", None),
    )
    if choice.finish_reason == "length":
        logger.warning(
            "DeepSeek response truncated feature=%s max_tokens=%s",
            feature, max_tokens)
    return choice.message.content


# langdetect's top-1 guess is unreliable on the short, mixed-script strings we
# get back (a Russian grammar note quoting English examples, an explanation under
# ~20 chars, etc.). Strict `detect(text) != expected` produced false positives —
# most damagingly Russian misread as bg/mk/uk — and every false positive cost a
# full second model call (doubling latency) AND prepended an ugly "[Language
# Error]" marker to correct, user-facing text. We now only flag a mismatch when
# we're confident: short strings pass, and the expected language is accepted if
# it appears anywhere in the probability ranking with a non-trivial share.
_LANG_MISMATCH_MIN_LEN = 25
_LANG_EXPECTED_MIN_PROB = 0.20


def _is_wrong_language(text: str, expected_code: str) -> bool:
    """True only when `text` is confidently NOT in `expected_code`. Conservative
    by design: when unsure, return False so we neither retry nor mark the text."""
    text = (text or "").strip()
    if len(text) < _LANG_MISMATCH_MIN_LEN:
        return False  # too short for reliable detection — trust the model
    try:
        ranked = detect_langs(text)  # list of "lang:prob", high→low
    except LangDetectException:
        return False
    detection_code = expected_code.replace("-", "_").split("_", 1)[0].lower()
    for guess in ranked:
        if guess.lang == detection_code and guess.prob >= _LANG_EXPECTED_MIN_PROB:
            return False
    # Expected language absent (or negligible) in the ranking → genuinely wrong.
    return True


async def parse_correction_response(
    response: str,
    interface_language_code: str,
    request: CorrectionRequest,
    client: AsyncOpenAI,
    retry_count: int = 0,
) -> Dict:
    result = _extract_json_object(response)

    required_fields = [
        "corrected_text", "error_analysis", "error_statistics", "explanation",
        "grammar_notes", "pronunciation_tips", "alternatives", "level_appropriate_suggestions",
    ]
    for field in required_fields:
        if field not in result:
            result[field] = [] if field == "error_analysis" else ""

    if not isinstance(result["error_analysis"], list):
        result["error_analysis"] = []

    valid_error_analysis = []
    required_error_fields = ["type", "original", "corrected", "explanation"]
    for error in result["error_analysis"]:
        if all(field in error for field in required_error_fields):
            valid_error_analysis.append(error)
    result["error_analysis"] = valid_error_analysis

    # pronunciation_tips is deliberately excluded: it mixes interface-language
    # prose with IPA and target-language words (e.g. "'student' [ˈstjuːdənt]"),
    # which langdetect reliably misreads — flagging it was always a false
    # positive. (It is also currently hidden in the client UI.)
    explanation_fields = ["explanation", "grammar_notes", "level_appropriate_suggestions"]
    language_mismatch = False
    for field in explanation_fields:
        text = result.get(field, "")
        if text and _is_wrong_language(text, interface_language_code):
            language_mismatch = True
            result[field] = f"[Language Error: Expected {interface_language_code}] {text}"

    for error in result["error_analysis"]:
        explanation = error.get("explanation", "")
        if explanation and _is_wrong_language(explanation, interface_language_code):
            language_mismatch = True
            error["explanation"] = f"[Language Error: Expected {interface_language_code}] {explanation}"

    if language_mismatch and retry_count < 1:
        logger.info(f"Retrying request due to language mismatch (attempt {retry_count + 1}/1)")
        retry_prompt = generate_teacher_prompt(request, retry=True)
        retry_text = await _call_deepseek(client, retry_prompt, request.text)
        return await parse_correction_response(
            retry_text,
            interface_language_code,
            request,
            client,
            retry_count=retry_count + 1,
        )
    elif language_mismatch:
        logger.warning("Language mismatch persists after retry, returning response with warnings")

    if isinstance(result["error_statistics"], dict):
        s = result["error_statistics"]
        result["error_statistics"] = (
            f"Grammar: {s.get('grammar', 0)}, Vocabulary: {s.get('vocabulary', 0)}, "
            f"Pronunciation: {s.get('pronunciation', 0)}, Other: {s.get('other', 0)}"
        )

    grammar_count = sum(1 for e in result["error_analysis"] if e.get("type") == "grammar")
    vocab_count = sum(1 for e in result["error_analysis"] if e.get("type") == "vocabulary")
    pron_count = sum(1 for e in result["error_analysis"] if e.get("type") == "pronunciation")
    other_count = sum(1 for e in result["error_analysis"] if e.get("type") == "other")
    result["error_statistics"] = f"Grammar: {grammar_count}, Vocabulary: {vocab_count}, Pronunciation: {pron_count}, Other: {other_count}"

    if isinstance(result["alternatives"], list):
        result["alternatives"] = "\n".join(
            f"{alt.get('sentence', '')}: {alt.get('explanation', '')}"
            for alt in result["alternatives"]
            if isinstance(alt, dict) and "sentence" in alt and "explanation" in alt
        )

    logger.info(f"Parsed response: corrected_text_length={len(result.get('corrected_text', ''))}, error_count={len(result.get('error_analysis', []))}")
    return result


rate_limiter = RateLimiter(max_requests=20, time_frame=timedelta(minutes=1))
quiz_ip_generation_limiter = RateLimiter(
    max_requests=3, time_frame=timedelta(minutes=1))
quiz_global_generation_limiter = RateLimiter(
    max_requests=60, time_frame=timedelta(minutes=1))
_quiz_generation_locks: Dict[str, asyncio.Lock] = {}
response_cache = ResponseCache()
llm_usage_meter = LLMUsageMeter()


@asynccontextmanager
async def _quiz_generation_lock(cache_key: str):
    """Deduplicate a cold quiz generation without retaining idle locks forever."""
    lock = _quiz_generation_locks.setdefault(cache_key, asyncio.Lock())
    try:
        async with lock:
            yield
    finally:
        # This exact lock is idle after the context exits. Do not remove a
        # replacement lock another coroutine may have installed meanwhile.
        if _quiz_generation_locks.get(cache_key) is lock and not lock.locked():
            _quiz_generation_locks.pop(cache_key, None)


@app.post("/process-text/")
async def process_text(
    request: Request,
    correction_request: CorrectionRequest,
    client: AuthenticatedClient = Depends(verify_client),
):
    if not rate_limiter.is_allowed(client.principal_id):
        raise HTTPException(status_code=429, detail="Too many requests")

    logger.info(
        "Request received auth=%s lang=%s level=%s style=%s",
        client.auth_scheme,
        correction_request.language,
        correction_request.level,
        correction_request.style,
    )

    # Reuse a stored result for an identical request — saves a model call.
    cache_key = response_cache.make_key(
        correction_request.text,
        correction_request.language,
        correction_request.level,
        correction_request.style,
        correction_request.interface_language,
        correction_request.context,
    )
    cached = response_cache.get(cache_key)
    if cached is not None:
        llm_usage_meter.record_cache_hit(feature="analysis", model=DEEPSEEK_MODEL)
        logger.info("Cache hit auth=%s", client.auth_scheme)
        return JSONResponse(content=cached, media_type="application/json; charset=utf-8")

    if not _deepseek_client:
        raise HTTPException(status_code=500, detail="DeepSeek API key not configured")

    try:
        if ANALYSIS_CONTRACT_VERSION == "v2":
            prompt, user_payload = build_v2_prompt(correction_request)
        else:
            prompt, user_payload = generate_teacher_prompt(correction_request), correction_request.text
        interface_lang_config = INTERFACE_LANGUAGES.get(correction_request.interface_language, INTERFACE_LANGUAGES["en"])

        sections = None
        last_parse_error: Optional[ValueError] = None
        for attempt in range(2):
            response_text = await _call_deepseek(
                _deepseek_client, prompt, user_payload, feature="analysis")
            try:
                sections = (
                    parse_v2_correction_response(response_text, correction_request)
                    if ANALYSIS_CONTRACT_VERSION == "v2"
                    else await parse_correction_response(
                        response_text, interface_lang_config["language_code"], correction_request, _deepseek_client
                    )
                )
                break
            except ValueError as e:
                last_parse_error = e
                logger.error(f"Attempt {attempt + 1}/2: invalid JSON from model: {e}")
        if sections is None:
            logger.error(f"Model returned unparseable JSON after retries: {last_parse_error}")
            raise HTTPException(status_code=502, detail="Language model returned an invalid response, please try again")

        content = {
            "corrected_text": sections["corrected_text"],
            "error_analysis": sections["error_analysis"],
            "error_statistics": sections["error_statistics"],
            "explanation": sections["explanation"],
            "grammar_notes": sections["grammar_notes"],
            "pronunciation_tips": sections["pronunciation_tips"],
            "alternatives": sections["alternatives"],
            "level_appropriate_suggestions": sections["level_appropriate_suggestions"],
            "context": correction_request.context or "",
        }
        response_cache.put(cache_key, content)
        return JSONResponse(
            content=content,
            media_type="application/json; charset=utf-8",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Processing error: %s", type(e).__name__, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


def _sse(event: str, data: Dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _build_content(sections: Dict, request: CorrectionRequest) -> Dict:
    return {
        "corrected_text": sections["corrected_text"],
        "error_analysis": sections["error_analysis"],
        "error_statistics": sections["error_statistics"],
        "explanation": sections["explanation"],
        "grammar_notes": sections["grammar_notes"],
        "pronunciation_tips": sections["pronunciation_tips"],
        "alternatives": sections["alternatives"],
        "level_appropriate_suggestions": sections["level_appropriate_suggestions"],
        "context": request.context or "",
    }


@app.post("/process-text-stream/")
async def process_text_stream(
    request: Request,
    correction_request: CorrectionRequest,
    client: AuthenticatedClient = Depends(verify_client),
):
    """Server-Sent Events variant of /process-text/. Emits the corrected text as
    soon as the model finishes it (the first, short field) so the client can show
    it after ~1-2s instead of waiting ~10s for the full analysis, then a final
    event with the complete result. Same request body + auth + cache as the
    non-streaming route, which stays for older clients.

    Events: `partial` {corrected_text}, then `result` {full analysis}, or `error`.
    """
    if not rate_limiter.is_allowed(client.principal_id):
        raise HTTPException(status_code=429, detail="Too many requests")

    cache_key = response_cache.make_key(
        correction_request.text, correction_request.language,
        correction_request.level, correction_request.style,
        correction_request.interface_language, correction_request.context,
    )
    interface_lang_config = INTERFACE_LANGUAGES.get(
        correction_request.interface_language, INTERFACE_LANGUAGES["en"])

    async def event_stream():
        # Cache hit → deliver the same two-event shape instantly.
        cached = response_cache.get(cache_key)
        if cached is not None:
            llm_usage_meter.record_cache_hit(
                feature="analysis_stream", model=DEEPSEEK_MODEL)
            yield _sse("partial", {"corrected_text": cached.get("corrected_text", "")})
            yield _sse("result", cached)
            return

        if not _deepseek_client:
            yield _sse("error", {"detail": "DeepSeek API key not configured"})
            return

        try:
            if ANALYSIS_CONTRACT_VERSION == "v2":
                prompt, user_payload = build_v2_prompt(correction_request)
            else:
                prompt, user_payload = generate_teacher_prompt(correction_request), correction_request.text
            buffer: List[str] = []
            emitted_partial = False
            async for delta in _call_deepseek_stream(
                    _deepseek_client, prompt, user_payload,
                    feature="analysis_stream"):
                buffer.append(delta)
                if not emitted_partial:
                    m = _CORRECTED_TEXT_RE.search("".join(buffer))
                    if m:
                        try:
                            corrected = json.loads(f'"{m.group(1)}"')
                        except json.JSONDecodeError:
                            corrected = m.group(1)
                        if corrected:
                            yield _sse("partial", {"corrected_text": corrected})
                            emitted_partial = True

            full_text = "".join(buffer)
            try:
                sections = (
                    parse_v2_correction_response(full_text, correction_request)
                    if ANALYSIS_CONTRACT_VERSION == "v2"
                    else await parse_correction_response(
                        full_text, interface_lang_config["language_code"],
                        correction_request, _deepseek_client)
                )
            except ValueError:
                yield _sse("error", {"detail": "Language model returned an invalid response, please try again"})
                return

            content = _build_content(sections, correction_request)
            response_cache.put(cache_key, content)
            yield _sse("result", content)
        except Exception as e:
            logger.error("Stream processing error: %s", type(e).__name__, exc_info=True)
            yield _sse("error", {"detail": "Internal server error"})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


_RULE_MEDIA = "application/json; charset=utf-8"


@app.get("/rule")
async def get_rule_endpoint(
    request: Request,
    learning: str,
    interface: str,
    rule_id: str,
    client: AuthenticatedClient = Depends(verify_client),
):
    """Full lesson for one rule. Served from the pre-generated static file when
    available; otherwise generated on demand for a valid taxonomy key and cached
    (the "grows with use" path). See rules/DYNAMIC_RULES_SPEC.md."""
    if not rate_limiter.is_allowed(client.principal_id):
        raise HTTPException(status_code=429, detail="Too many requests")
    # 1) static pre-generated content
    try:
        rule = rules_store.get_rule(learning, interface, rule_id)
        return JSONResponse(content=rule, media_type=_RULE_MEDIA)
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid parameters")
    except (rules_store.RulesNotFound, rules_store.RuleNotFound):
        pass  # fall through to on-demand generation

    # 2) rule_id MUST be a known taxonomy key — never mint ids from free text
    title = rules_store.topic_title(learning, rule_id)
    if not title:
        raise HTTPException(status_code=404, detail="rule not found")

    # 3) on-demand cache
    cache_key = f"rule::{learning}::{interface}::{rule_id}"
    cached = response_cache.get(cache_key)
    if cached is not None:
        llm_usage_meter.record_cache_hit(feature="rule", model=DEEPSEEK_MODEL)
        return JSONResponse(content=cached, media_type=_RULE_MEDIA)

    # 4) lazy generation via the model, then cache
    if not _deepseek_client:
        raise HTTPException(status_code=503, detail="generation unavailable")
    learning_name = catalog_display_name(LANGUAGE_CATALOG, learning)
    interface_name = catalog_display_name(LANGUAGE_CATALOG, interface)
    prompt = rules_store.build_rule_prompt(title, learning_name, interface_name)
    try:
        raw = await _call_deepseek(
            _deepseek_client, prompt, "Generate the lesson as JSON now.")
        data = rules_store.extract_json(raw)
        if not data.get("title") or not data.get("explanation"):
            raise ValueError("incomplete rule")
        rule = {
            "rule_id": rule_id,
            "topic": title,
            "title": data["title"],
            "explanation": data["explanation"],
            "examples": data.get("examples", []),
            "exercises": data.get("exercises", []),
        }
    except Exception as e:
        logger.error(
            f"Rule gen failed {learning}/{interface}/{rule_id}: {type(e).__name__}")
        raise HTTPException(status_code=502, detail="rule generation failed")
    response_cache.put(cache_key, rule)
    return JSONResponse(content=rule, media_type=_RULE_MEDIA)


@app.get("/rule-quiz")
async def get_rule_quiz_endpoint(
    request: Request,
    learning: str,
    interface: str,
    rule_id: str,
    level: str,
    question_count: int = 5,
    client: AuthenticatedClient = Depends(verify_client),
):
    """Return a cached or lazily generated quiz at the requested CEFR level."""
    if not rate_limiter.is_allowed(client.principal_id):
        raise HTTPException(status_code=429, detail="Too many requests")

    try:
        quiz_level, quiz_question_count = rules_store.validate_quiz_parameters(
            learning, interface, rule_id, level, question_count)
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid parameters")

    # Syntactically valid but unsupported codes must fail before touching the
    # shared cache or invoking the paid model endpoint.
    if learning not in LANGUAGE_CONFIGS or interface not in INTERFACE_LANGUAGES:
        raise HTTPException(status_code=400, detail="unsupported language")

    topics = rules_store.load_topics(learning)
    if not topics:
        raise HTTPException(status_code=404, detail="learning taxonomy not found")
    title = next(
        (topic.get("title") for topic in topics
         if topic.get("rule_id") == rule_id),
        None,
    )
    if not isinstance(title, str) or not title.strip():
        raise HTTPException(status_code=404, detail="rule not found")

    cache_key = (
        f"rule-quiz::v2::{learning}::{interface}::{rule_id}::{quiz_level}::{quiz_question_count}")
    cached = response_cache.get(cache_key)
    if cached is not None:
        llm_usage_meter.record_cache_hit(feature="rule_quiz", model=DEEPSEEK_MODEL)
        try:
            clean = rules_store.validate_rule_quiz(
                cached, expected_question_count=quiz_question_count)
            result = {
                "rule_id": rule_id,
                "learning": learning,
                "interface": interface,
                "level_band": quiz_level,
                "questions": clean["questions"],
            }
            return JSONResponse(content=result, media_type=_RULE_MEDIA)
        except ValueError:
            logger.warning(
                "Ignoring invalid rule quiz cache entry %s/%s/%s/%s",
                learning, interface, rule_id, quiz_level)

    # Single-flight by quiz key: concurrent first opens await one generation
    # instead of multiplying paid model calls for the same content.
    async with _quiz_generation_lock(cache_key):
        cached = response_cache.get(cache_key)
        if cached is not None:
            llm_usage_meter.record_cache_hit(
                feature="rule_quiz", model=DEEPSEEK_MODEL)
            try:
                clean = rules_store.validate_rule_quiz(
                    cached, expected_question_count=quiz_question_count)
                return JSONResponse(content={
                    "rule_id": rule_id,
                    "learning": learning,
                    "interface": interface,
                    "level_band": quiz_level,
                    "questions": clean["questions"],
                }, media_type=_RULE_MEDIA)
            except ValueError:
                pass

        # A new anonymous Firebase UID must not reset the paid-generation
        # budget. Cache hits remain free and do not consume these quotas.
        client_ip = _get_client_ip(request)
        if (not quiz_ip_generation_limiter.is_allowed(client_ip)
                or not quiz_global_generation_limiter.is_allowed("global")):
            raise HTTPException(
                status_code=429, detail="Quiz generation limit reached")
        if not _deepseek_client:
            raise HTTPException(status_code=503, detail="generation unavailable")

        rule_context = None
        try:
            rule_context = rules_store.get_rule(learning, interface, rule_id)
        except (rules_store.RulesNotFound, rules_store.RuleNotFound):
            pass

        prompt = rules_store.build_rule_quiz_prompt(
            title=title,
            learning_name=catalog_display_name(LANGUAGE_CATALOG, learning),
            interface_name=catalog_display_name(LANGUAGE_CATALOG, interface),
            level=quiz_level,
            question_count=quiz_question_count,
            rule_context=rule_context,
        )
        clean = None
        for attempt in range(2):
            try:
                raw = await _call_deepseek(
                    _deepseek_client,
                    prompt,
                    "Generate the quiz JSON now.",
                    feature="rule_quiz",
                    max_tokens=1800,
                )
            except Exception as error:
                logger.error(
                    "Rule quiz model call failed %s/%s/%s/%s: %s",
                    learning, interface, rule_id, quiz_level,
                    type(error).__name__)
                raise HTTPException(
                    status_code=502, detail="quiz generation failed")

            try:
                if not isinstance(raw, str):
                    raise ValueError("quiz response must be text")
                clean = rules_store.validate_rule_quiz(
                    rules_store.extract_json(raw),
                    expected_question_count=quiz_question_count)
                break
            except (TypeError, ValueError):
                logger.warning(
                    "Invalid rule quiz response %s/%s/%s/%s attempt=%s",
                    learning, interface, rule_id, quiz_level, attempt + 1)

        if clean is None:
            raise HTTPException(status_code=502, detail="quiz generation failed")

        result = {
            "rule_id": rule_id,
            "learning": learning,
            "interface": interface,
            "level_band": quiz_level,
            "questions": clean["questions"],
        }
        response_cache.put(cache_key, result)
        return JSONResponse(content=result, media_type=_RULE_MEDIA)


class ResolveRuleRequest(BaseModel):
    # Legacy text fields are accepted only so already-shipped clients get a
    # harmless unresolved result during migration.  They are never embedded in
    # a prompt or used for keyword matching.
    learning: str = Field(max_length=16)
    interface: str = Field(default="", max_length=16)
    concept_code: str = Field(default="", max_length=100)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    type: str = Field(default="", max_length=32)
    original: str = Field(default="", max_length=300)
    corrected: str = Field(default="", max_length=300)
    explanation: str = Field(default="", max_length=600)


@app.post("/resolve-rule")
async def resolve_rule_endpoint(
    request: Request,
    body: ResolveRuleRequest,
    client: AuthenticatedClient = Depends(verify_client),
):
    """Resolve a V2 concept to one exact active rule, or return unresolved.

    There is deliberately no text-to-rule fallback here: returning an unrelated
    lesson is worse than showing the rule list.  New clients send concept_code
    from the analysis result; legacy clients receive null until upgraded.
    """
    if not rate_limiter.is_allowed(client.principal_id):
        raise HTTPException(status_code=429, detail="Too many requests")
    learning = body.learning
    try:
        topics = rules_store.load_topics(learning)
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid parameters")
    if not topics:
        return JSONResponse(content={"rule_id": None}, media_type=_RULE_MEDIA)
    threshold = float(os.getenv("RULE_MAPPING_MIN_CONFIDENCE", "0.75"))
    resolved = (
        rules_store.resolve_concept(learning, body.concept_code)
        if body.concept_code and body.confidence >= threshold
        else None
    )
    return JSONResponse(content={"rule_id": resolved}, media_type=_RULE_MEDIA)


@app.get("/health")
async def health_check():
    try:
        if not _deepseek_client:
            return JSONResponse(
                status_code=503,
                content={"status": "unhealthy", "reason": "DeepSeek API key not configured", "timestamp": datetime.now().isoformat()},
            )
        required_files = ["languages/catalog.v2.json"]
        for f in required_files:
            if not os.path.exists(f):
                return JSONResponse(
                    status_code=503,
                    content={"status": "unhealthy", "reason": f"Configuration file {f} not found", "timestamp": datetime.now().isoformat()},
                )
        return JSONResponse(content={"status": "healthy", "timestamp": datetime.now().isoformat(), "version": APP_VERSION})
    except Exception:
        logger.error("Health check error", exc_info=True)
        return JSONResponse(status_code=503, content={"status": "unhealthy", "timestamp": datetime.now().isoformat()})


@app.get("/")
async def root():
    return JSONResponse(content={"message": "Speech Correction API is running", "version": APP_VERSION})


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = datetime.now()
    response = await call_next(request)
    process_time = (datetime.now() - start_time).total_seconds()
    app_check_status = getattr(request.state, "app_check_status", "not_checked")
    logger.info(
        "%s %s - %s, app_check=%s, Process time: %.4fs",
        request.method,
        request.url.path,
        response.status_code,
        app_check_status,
        process_time,
    )
    return response


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8081))
    uvicorn.run(app, host="0.0.0.0", port=port)
