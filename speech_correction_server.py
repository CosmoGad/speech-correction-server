import os
import json
import logging
import asyncio
import hashlib
import sqlite3
from datetime import datetime, timedelta
from functools import lru_cache
import threading

import regex as re
from fastapi import FastAPI, HTTPException, Request, Depends, Security
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, Dict, List
from langdetect import detect_langs, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

from openai import AsyncOpenAI, APIError, RateLimitError
from dotenv import load_dotenv
from logging.handlers import RotatingFileHandler

DetectorFactory.seed = 0
load_dotenv()

APP_VERSION = "2.2.0"

# Output budget for the model. Long texts with many errors produce large JSON
# payloads (Cyrillic explanations are token-expensive); 1500 used to truncate
# them mid-document, which surfaced to clients as "Error processing response".
DEEPSEEK_MAX_TOKENS = 4000
DEEPSEEK_MODEL = "deepseek-v4-flash"

# Precompile dangerous input patterns once at module load
_DANGEROUS_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [
    r"<script\b[^>]*>", r"</script>", r"javascript:", r"eval\(", r"expression\(", r"on\w+\s*=",
    r"\{\{.*?\}\}", r"\$\{.*?\}", r"\$\(.*\)", r"`.*?`",
    r"\.\./", r"\.\.\\", r"%2e%2e", r"%252e",
    r"[‮‎‏‪‫‬‭]",
]]
_VALID_TEXT_RE = re.compile(r"^[\p{L}\p{N}\s\.,!?()'\-]+$", re.UNICODE)

# Initialize DeepSeek async client
_deepseek_api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
_deepseek_client: Optional[AsyncOpenAI] = None
if _deepseek_api_key:
    _deepseek_client = AsyncOpenAI(
        api_key=_deepseek_api_key,
        base_url="https://api.deepseek.com/v1",
        timeout=30.0,
    )

# Client API key auth
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
_server_api_key = os.getenv("API_KEY")


async def verify_api_key(key: str = Security(_api_key_header)) -> None:
    if not _server_api_key:
        return  # API_KEY not set — open mode (dev/migration)
    if key != _server_api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# Load static configurations at startup
with open("language_configs.json", "r", encoding="utf-8") as f:
    LANGUAGE_CONFIGS = json.load(f)

with open("level_details.json", "r", encoding="utf-8") as f:
    LEVEL_DETAILS = json.load(f)

with open("interface_languages.json", "r", encoding="utf-8") as f:
    INTERFACE_LANGUAGES = json.load(f)

with open("context_instructions.json", "r", encoding="utf-8") as f:
    CONTEXT_INSTRUCTIONS = json.load(f)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
handler = RotatingFileHandler("app.log", maxBytes=10000000, backupCount=5, encoding="utf-8")
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

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
# in rules/ (see rules/README.md). Same X-API-Key auth as the other routes.
from rules_api import router as rules_router
import rules_store

app.include_router(rules_router, dependencies=[Depends(verify_api_key)])


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
    """Content-addressed cache of analysis results in SQLite. Identical
    requests (same text + language + level + style + interface + context) reuse
    a stored result instead of calling the model again — saving tokens on the
    repeated mistakes many users make. Keyed by content only (no user id), with
    a TTL, so it never builds a per-user profile."""

    def __init__(self, db_path: str = "response_cache.db", ttl_days: int = 30):
        self.db_path = db_path
        self.ttl = timedelta(days=ttl_days)
        self._lock = threading.Lock()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS cache ("
                "key TEXT PRIMARY KEY, response TEXT NOT NULL, created_at TEXT NOT NULL)"
            )

    @staticmethod
    def make_key(text, language, level, style, interface_language, context) -> str:
        raw = json.dumps(
            [text, language, level, style, interface_language, context or ""],
            ensure_ascii=False,
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[Dict]:
        with self._lock, sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT response, created_at FROM cache WHERE key = ?", (key,)
            ).fetchone()
            if not row:
                return None
            response, created_at = row
            try:
                fresh = datetime.now() - datetime.fromisoformat(created_at) <= self.ttl
            except ValueError:
                fresh = False
            if not fresh:
                conn.execute("DELETE FROM cache WHERE key = ?", (key,))
                return None
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                return None

    def put(self, key: str, response: Dict) -> None:
        try:
            with self._lock, sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO cache (key, response, created_at) "
                    "VALUES (?, ?, ?)",
                    (key, json.dumps(response, ensure_ascii=False), datetime.now().isoformat()),
                )
        except sqlite3.Error as e:
            # A cache write must never break a successful response.
            logger.error(f"Cache write failed: {e}")


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
            level_description=level_info["description"].get(interface_lang_config["language_code"], level_info["description"]["English"]),
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


async def _call_deepseek(client: AsyncOpenAI, prompt: str, user_text: str) -> str:
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
        max_tokens=DEEPSEEK_MAX_TOKENS,
        response_format={"type": "json_object"},
    )
    choice = response.choices[0]
    if choice.finish_reason == "length":
        logger.warning("DeepSeek response truncated at max_tokens=%s", DEEPSEEK_MAX_TOKENS)
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
    for guess in ranked:
        if guess.lang == expected_code and guess.prob >= _LANG_EXPECTED_MIN_PROB:
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
response_cache = ResponseCache()


@app.post("/process-text/")
async def process_text(
    request: Request,
    correction_request: CorrectionRequest,
    _: None = Depends(verify_api_key),
):
    client_ip = _get_client_ip(request)

    if not rate_limiter.is_allowed(client_ip):
        raise HTTPException(status_code=429, detail="Too many requests")

    logger.info(f"Request received from {client_ip}, lang={correction_request.language}, level={correction_request.level}, style={correction_request.style}")

    # Reuse a stored result for an identical request — saves a model call.
    cache_key = ResponseCache.make_key(
        correction_request.text,
        correction_request.language,
        correction_request.level,
        correction_request.style,
        correction_request.interface_language,
        correction_request.context,
    )
    cached = response_cache.get(cache_key)
    if cached is not None:
        logger.info(f"Cache hit for {client_ip}")
        return JSONResponse(content=cached, media_type="application/json; charset=utf-8")

    if not _deepseek_client:
        raise HTTPException(status_code=500, detail="DeepSeek API key not configured")

    try:
        prompt = generate_teacher_prompt(correction_request)
        interface_lang_config = INTERFACE_LANGUAGES.get(correction_request.interface_language, INTERFACE_LANGUAGES["en"])

        sections = None
        last_parse_error: Optional[ValueError] = None
        for attempt in range(2):
            response_text = await _call_deepseek(_deepseek_client, prompt, correction_request.text)
            try:
                sections = await parse_correction_response(
                    response_text, interface_lang_config["language_code"], correction_request, _deepseek_client
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
            "original_text": correction_request.text,
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
        logger.error(f"Processing error for {client_ip}: {type(e).__name__}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


_RULE_MEDIA = "application/json; charset=utf-8"


@app.get("/rule")
async def get_rule_endpoint(
    request: Request,
    learning: str,
    interface: str,
    rule_id: str,
    _: None = Depends(verify_api_key),
):
    """Full lesson for one rule. Served from the pre-generated static file when
    available; otherwise generated on demand for a valid taxonomy key and cached
    (the "grows with use" path). See rules/DYNAMIC_RULES_SPEC.md."""
    if not rate_limiter.is_allowed(_get_client_ip(request)):
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
        return JSONResponse(content=cached, media_type=_RULE_MEDIA)

    # 4) lazy generation via the model, then cache
    if not _deepseek_client:
        raise HTTPException(status_code=503, detail="generation unavailable")
    learning_name = rules_store.LANGUAGE_NAMES.get(learning, learning)
    interface_name = rules_store.LANGUAGE_NAMES.get(interface, interface)
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


class ResolveRuleRequest(BaseModel):
    # Length caps mirror the validation on CorrectionRequest: these fields are
    # embedded into the resolve prompt, so bounding them keeps a caller from
    # inflating token cost with oversized payloads. (The picked rule_id is also
    # validated against the fixed taxonomy, so the output can't be injected.)
    learning: str = Field(max_length=16)
    interface: str = Field(default="", max_length=16)
    type: str = Field(default="", max_length=32)
    original: str = Field(default="", max_length=300)
    corrected: str = Field(default="", max_length=300)
    explanation: str = Field(default="", max_length=600)


@app.post("/resolve-rule")
async def resolve_rule_endpoint(
    request: Request,
    body: ResolveRuleRequest,
    _: None = Depends(verify_api_key),
):
    """Map a correction error to the best-matching rule_id from the fixed
    taxonomy (or null). The model can only SELECT an existing id, never invent
    one — the anti-duplication guarantee. POST keeps the user's text out of
    request URLs/logs; cached per error signature; rate-limited."""
    if not rate_limiter.is_allowed(_get_client_ip(request)):
        raise HTTPException(status_code=429, detail="Too many requests")
    learning = body.learning
    err_type = body.type
    original = body.original
    corrected = body.corrected
    explanation = body.explanation
    try:
        topics = rules_store.load_topics(learning)
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid parameters")
    if not topics:
        return JSONResponse(content={"rule_id": None}, media_type=_RULE_MEDIA)

    sig = rules_store.error_signature(learning, err_type, original, corrected)
    cache_key = f"resolve::{sig}"
    cached = response_cache.get(cache_key)
    if cached is not None:
        return JSONResponse(content=cached, media_type=_RULE_MEDIA)

    # No model configured → return null WITHOUT caching (so it resolves once
    # generation is available again).
    if not _deepseek_client:
        return JSONResponse(content={"rule_id": None}, media_type=_RULE_MEDIA)

    valid_ids = {t["rule_id"] for t in topics}
    prompt = rules_store.build_resolve_prompt(
        rules_store.LANGUAGE_NAMES.get(learning, learning), topics,
        err_type, original, corrected, explanation)
    try:
        raw = await _call_deepseek(
            _deepseek_client, prompt, "Return the JSON now.")
        picked = rules_store.extract_json(raw).get("rule_id")
    except Exception as e:
        # Transient failure → return null but DON'T cache it (avoid poisoning
        # this error with a permanent "no rule" for the cache TTL).
        logger.error(f"Rule resolve failed {learning}: {type(e).__name__}")
        return JSONResponse(content={"rule_id": None}, media_type=_RULE_MEDIA)

    # Definitive answer (a valid pick, or a genuine "none") — safe to cache.
    result = {"rule_id": picked if (isinstance(picked, str)
                                    and picked in valid_ids) else None}
    response_cache.put(cache_key, result)
    return JSONResponse(content=result, media_type=_RULE_MEDIA)


@app.get("/health")
async def health_check():
    try:
        if not _deepseek_client:
            return JSONResponse(
                status_code=503,
                content={"status": "unhealthy", "reason": "DeepSeek API key not configured", "timestamp": datetime.now().isoformat()},
            )
        required_files = ["language_configs.json", "level_details.json", "interface_languages.json", "context_instructions.json"]
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
    logger.info(f"{request.method} {request.url.path} - {response.status_code}, Process time: {process_time:.4f}s")
    return response


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8081))
    uvicorn.run(app, host="0.0.0.0", port=port)
