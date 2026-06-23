#!/usr/bin/env python3
"""Offline grammar-rules generator for the "rule book" feature.

Pre-generates short lessons (rule + examples + exercises) for the most common
mistakes of a learning language, written in a given interface language, and
saves them as static JSON. The production server then serves these for free
(no model call), falling back to on-demand generation only for rules that were
not pre-generated.

The seed topics come from ``language_configs.json`` (``common_errors`` +
``pronunciation_focus``), so no corpus of real user errors is needed.

Provider is configurable so you can run this against a *free* endpoint
(e.g. an OpenModel promo key) to save tokens, while production keeps using
the paid DeepSeek key:

    RULES_API_KEY   API key            (falls back to DEEPSEEK_API_KEY)
    RULES_API_BASE  OpenAI-compatible base URL
                                       (default https://api.deepseek.com/v1)
    RULES_MODEL     model name         (default deepseek-v4-flash)
    RULES_API_MODE  "chat" | "responses" | "messages"  (default "chat").
                    "messages" = Anthropic Messages API (/v1/messages) — this is
                    how OpenModel serves DeepSeek (model deepseek-v4-flash).

Usage:
    python tools/generate_rules.py --learning en --interface en
    python tools/generate_rules.py --learning el --interface ru --limit 3
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    sys.exit("The 'openai' package is required: pip install openai")

try:
    import httpx
except ImportError:
    sys.exit("The 'httpx' package is required: pip install httpx")

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # .env loading is optional

SERVER_DIR = Path(__file__).resolve().parent.parent
RULES_DIR = SERVER_DIR / "rules"

# Learning-language code -> English display name (mirrors the client's
# LanguageUtils.getLanguageName). Used only to phrase the prompt clearly.
LANGUAGE_NAMES = {
    "en": "English", "ru": "Russian", "de": "German", "fr": "French",
    "es": "Spanish", "it": "Italian", "pt": "Portuguese", "nl": "Dutch",
    "pl": "Polish", "uk": "Ukrainian", "tr": "Turkish", "ar": "Arabic",
    "fa": "Persian", "ur": "Urdu", "sr": "Serbian", "sv": "Swedish",
    "da": "Danish", "no": "Norwegian", "el": "Greek", "cs": "Czech",
    "ro": "Romanian", "hu": "Hungarian", "bg": "Bulgarian", "hr": "Croatian",
    "sk": "Slovak", "sl": "Slovenian", "fi": "Finnish", "lt": "Lithuanian",
    "lv": "Latvian", "et": "Estonian",
}


def slugify(text: str) -> str:
    """Deterministic ASCII-ish slug for a topic (rule_id stem)."""
    text = text.strip().lower()
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    text = re.sub(r"[\s_-]+", "-", text)
    return text.strip("-") or "rule"


def extract_json(raw: str) -> dict:
    """Tolerant JSON extraction: strips code fences and surrounding prose."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(raw[start : end + 1])
        raise


def extract_json_array(raw: str) -> list:
    """Tolerant JSON-array extraction (for the topic list)."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("["), raw.rfind("]")
        if start != -1 and end != -1 and end > start:
            return json.loads(raw[start : end + 1])
        raise


def build_prompt(topic: str, learning_name: str, interface_name: str) -> str:
    return (
        f"You are an expert {learning_name} teacher creating a short lesson "
        f"for a learner whose language is {interface_name}.\n\n"
        f"Topic (a common {learning_name} mistake): \"{topic}\".\n\n"
        "Produce a compact, practical micro-lesson. Return ONLY a valid JSON "
        "object (UTF-8, no markdown, no commentary) with this exact shape:\n"
        "{\n"
        '  "title": string,            // short lesson title\n'
        '  "explanation": string,      // 2-4 sentences explaining the rule\n'
        '  "examples": [               // 2-3 items\n'
        '    {"wrong": string, "right": string, "note": string}\n'
        "  ],\n"
        '  "exercises": [              // 2-3 items\n'
        '    {"prompt": string, "answer": string}\n'
        "  ]\n"
        "}\n\n"
        f"LANGUAGE RULES (strict):\n"
        f"- title, explanation, note and exercise prompts: in {interface_name}.\n"
        f"- wrong, right and exercise answers (the actual language samples): "
        f"in {learning_name}.\n"
        "- Keep it concise and beginner-friendly."
    )


def _extract_responses_text(payload: dict) -> str:
    """Pulls the text out of an OpenAI Responses API payload, tolerating a
    gateway envelope ({success, data, error}) around the real response."""
    if isinstance(payload.get("data"), dict) and "output" in payload["data"]:
        payload = payload["data"]
    if payload.get("output_text"):
        return payload["output_text"]
    chunks = []
    for item in payload.get("output", []):
        for c in item.get("content", []):
            if c.get("type") in ("output_text", "text") and c.get("text"):
                chunks.append(c["text"])
    return "\n".join(chunks)


def _extract_messages_text(payload: dict) -> str:
    """Pulls assistant text out of an Anthropic Messages API payload,
    ignoring 'thinking' blocks. (OpenModel serves DeepSeek via this protocol.)"""
    if isinstance(payload.get("data"), dict) and "content" in payload["data"]:
        payload = payload["data"]
    chunks = []
    for block in payload.get("content", []):
        if block.get("type") == "text" and block.get("text"):
            chunks.append(block["text"])
    return "\n".join(chunks)


def _call_model(client: OpenAI, model: str, mode: str, prompt: str,
                base_url: str, api_key: str) -> str:
    """Single completion call across three provider shapes:
    - "chat":      OpenAI Chat Completions (default; paid DeepSeek)
    - "responses": OpenAI Responses API
    - "messages":  Anthropic Messages API (OpenModel serves DeepSeek here)
    The HTTP branches don't rely on the openai SDK version."""
    if mode == "messages":
        r = httpx.post(
            f"{base_url.rstrip('/')}/messages",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                # Generous budget: this model emits long "thinking" blocks that
                # otherwise eat the limit before the actual answer is produced.
                "max_tokens": 5000,
                "temperature": 0.4,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=150,
        )
        r.raise_for_status()
        text = _extract_messages_text(r.json())
        if not text:
            raise ValueError(f"empty response body: {r.text[:300]}")
        return text
    if mode == "responses":
        r = httpx.post(
            f"{base_url.rstrip('/')}/responses",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "input": prompt,
                "temperature": 0.4,
                "max_output_tokens": 1200,
            },
            timeout=90,
        )
        r.raise_for_status()
        text = _extract_responses_text(r.json())
        if not text:
            raise ValueError(f"empty response body: {r.text[:300]}")
        return text
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=1200,
    )
    return resp.choices[0].message.content or ""


def generate_rule(client: OpenAI, model: str, mode: str, topic: str,
                  learning_name: str, interface_name: str,
                  base_url: str, api_key: str) -> dict | None:
    prompt = build_prompt(topic, learning_name, interface_name)
    for attempt in range(2):
        try:
            content = _call_model(
                client, model, mode, prompt, base_url, api_key)
            data = extract_json(content)
            # Minimal validation
            if not data.get("title") or not data.get("explanation"):
                raise ValueError("missing title/explanation")
            data.setdefault("examples", [])
            data.setdefault("exercises", [])
            return data
        except Exception as exc:  # noqa: BLE001 - batch tool, log & retry
            print(f"    ! attempt {attempt + 1} failed: {exc}", file=sys.stderr)
            time.sleep(1.5)
    return None


def _norm_title(t: str) -> str:
    """Normalized form for duplicate detection (case/space/punct-insensitive)."""
    return re.sub(r"[^\w]+", " ", t, flags=re.UNICODE).lower().strip()


def load_registry(learning: str, out_dir: Path) -> list[dict]:
    """The topic registry is the single source of truth for rule_ids. It is
    interface-independent (a rule_id identifies the grammar concept, so the
    same id is reused across all interface languages). Lives in
    rules/topics_<learning>.json as [{rule_id, title}]."""
    path = out_dir / f"topics_{learning}.json"
    if path.is_file():
        return json.loads(path.read_text("utf-8")).get("topics", [])
    return []


def save_registry(learning: str, out_dir: Path, topics: list[dict]) -> None:
    path = out_dir / f"topics_{learning}.json"
    path.write_text(
        json.dumps({"learning": learning, "topics": topics},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def add_topics(registry: list[dict], new_titles: list[str]) -> int:
    """Append new topics, skipping duplicates by rule_id AND normalized title.
    Returns how many were actually added."""
    seen_ids = {t["rule_id"] for t in registry}
    seen_norm = {_norm_title(t["title"]) for t in registry}
    added = 0
    for title in new_titles:
        title = title.strip()
        if not title:
            continue
        norm = _norm_title(title)
        rid = slugify(title)
        if rid in seen_ids or norm in seen_norm:
            continue
        # Resolve rare slug collisions on genuinely different titles.
        base_id, n = rid, 2
        while rid in seen_ids:
            rid, n = f"{base_id}-{n}", n + 1
        registry.append({"rule_id": rid, "title": title})
        seen_ids.add(rid)
        seen_norm.add(norm)
        added += 1
    return added


def build_topics_prompt(learning_name: str, count: int,
                        existing_titles: list[str]) -> str:
    existing = "\n".join(f"- {t}" for t in existing_titles) or "(none yet)"
    return (
        f"You are an expert {learning_name} teacher. List the {count} most "
        f"common, distinct grammar/usage/pronunciation mistakes that learners "
        f"of {learning_name} make and that deserve their own short lesson.\n\n"
        f"Already covered (DO NOT repeat or rephrase these):\n{existing}\n\n"
        "Return ONLY a JSON array of short English topic strings (3-6 words "
        "each), e.g. [\"Article usage a/an/the\", \"Present perfect vs past "
        "simple\"]. Each must be a genuinely DIFFERENT concept from the others "
        "and from the already-covered list. No numbering, no commentary."
    )


def generate_topics(client: OpenAI, model: str, mode: str, base_url: str,
                    api_key: str, learning_name: str, count: int,
                    existing_titles: list[str]) -> list[str]:
    prompt = build_topics_prompt(learning_name, count, existing_titles)
    for attempt in range(2):
        try:
            content = _call_model(
                client, model, mode, prompt, base_url, api_key)
            data = extract_json_array(content)
            return [str(x) for x in data if str(x).strip()]
        except Exception as exc:  # noqa: BLE001
            print(f"  ! topic gen attempt {attempt + 1} failed: {exc}",
                  file=sys.stderr)
            time.sleep(1.5)
    return []


def main() -> int:
    # Windows consoles default to cp1251/cp437, which crashes on printing
    # umlauts/accents in topic titles. Force UTF-8 output.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001 - older/odd streams: just skip
            pass

    ap = argparse.ArgumentParser(description="Generate the rule book JSON.")
    ap.add_argument("--learning", required=True, help="learning language code")
    ap.add_argument("--interface", required=True, help="interface language code")
    ap.add_argument("--gen-topics", type=int, default=0,
                    help="ask the model for N new deduped topics first")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap NEW rules generated this run (0 = all missing)")
    ap.add_argument("--out", default=str(RULES_DIR), help="output directory")
    args = ap.parse_args()

    api_key = os.getenv("RULES_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        return _die("Set RULES_API_KEY (or DEEPSEEK_API_KEY) in env/.env")
    base_url = os.getenv("RULES_API_BASE", "https://api.deepseek.com/v1")
    model = os.getenv("RULES_MODEL", "deepseek-v4-flash")
    mode = os.getenv("RULES_API_MODE", "chat").lower()

    configs = json.loads((SERVER_DIR / "language_configs.json").read_text("utf-8"))
    cfg = configs.get(args.learning)
    if not cfg:
        return _die(f"No language_configs entry for '{args.learning}'")

    learning_name = LANGUAGE_NAMES.get(args.learning, args.learning)
    interface_name = LANGUAGE_NAMES.get(args.interface, args.interface)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    client = OpenAI(api_key=api_key, base_url=base_url)

    # 1) Topic registry = single source of truth for rule_ids (dedup lives here).
    registry = load_registry(args.learning, out_dir)
    if not registry:
        seed = list(cfg.get("common_errors", [])) + \
            list(cfg.get("pronunciation_focus", []))
        add_topics(registry, seed)
        save_registry(args.learning, out_dir, registry)
        print(f"Seeded registry with {len(registry)} topics from configs.")

    # 2) Optionally grow the registry with deduped, model-proposed topics.
    if args.gen_topics > 0:
        existing = [t["title"] for t in registry]
        proposed = generate_topics(
            client, model, mode, base_url, api_key,
            learning_name, args.gen_topics, existing)
        added = add_topics(registry, proposed)
        save_registry(args.learning, out_dir, registry)
        print(f"Topic registry: +{added} new (now {len(registry)} total).")

    if not registry:
        return _die(f"No topics for '{args.learning}'")

    # 3) Load existing content for this interface; only generate missing ids.
    out_path = out_dir / f"rules_{args.learning}_{args.interface}.json"
    existing_rules: dict[str, dict] = {}
    if out_path.is_file():
        prev = json.loads(out_path.read_text("utf-8"))
        existing_rules = {r["rule_id"]: r for r in prev.get("rules", [])}

    missing = [t for t in registry if t["rule_id"] not in existing_rules]
    if args.limit > 0:
        missing = missing[: args.limit]
    print(f"{learning_name} -> {interface_name}: {len(existing_rules)} existing, "
          f"{len(missing)} to generate (model={model}, mode={mode})")

    generated = 0
    for i, topic in enumerate(missing, 1):
        print(f"  [{i}/{len(missing)}] {topic['title']}")
        data = generate_rule(
            client, model, mode, topic["title"], learning_name, interface_name,
            base_url, api_key)
        if not data:
            print("    -> skipped (generation failed)", file=sys.stderr)
            continue
        existing_rules[topic["rule_id"]] = {
            "rule_id": topic["rule_id"],
            "topic": topic["title"],
            "title": data["title"],
            "explanation": data["explanation"],
            "examples": data["examples"],
            "exercises": data["exercises"],
        }
        generated += 1

    # Preserve registry order in the output.
    ordered = [existing_rules[t["rule_id"]] for t in registry
               if t["rule_id"] in existing_rules]
    payload = {
        "learning": args.learning,
        "interface": args.interface,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "rules": ordered,
    }
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Generated {generated} new; wrote {len(ordered)} rules -> {out_path}")
    return 0


def _die(msg: str) -> int:
    print(f"error: {msg}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
