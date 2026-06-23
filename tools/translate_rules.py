#!/usr/bin/env python3
"""Translate a rule set's interface-language fields into another interface
language, delegated to a cheap model (DeepSeek via OpenModel). Reuses the
provider config + helpers from generate_rules.py.

What gets translated (interface-language fields):
  title, explanation, each example's `note`, each exercise's `prompt`.
What is kept verbatim (learning-language / identity fields):
  rule_id, topic, example.wrong, example.right, exercise.answer.

Idempotent: only rules missing from the target file are translated, so reruns
fill gaps without duplicating. Source defaults to the English interface set.

Env (same as generate_rules.py): RULES_API_KEY / RULES_API_BASE / RULES_MODEL /
RULES_API_MODE.

Usage:
  python tools/translate_rules.py --learning en --target de
  python tools/translate_rules.py --learning fr --source en --target it --limit 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    sys.exit("The 'openai' package is required: pip install openai")

# Reuse provider plumbing + helpers from the generator (same tools/ dir).
from generate_rules import (  # noqa: E402
    LANGUAGE_NAMES,
    RULES_DIR,
    _call_model,
    extract_json,
)


def build_translate_prompt(payload: dict, source_name: str,
                           target_name: str) -> str:
    return (
        f"Translate the VALUES of this JSON from {source_name} into "
        f"{target_name}. Keep the meaning, keep it natural and concise, and "
        f"keep the EXACT same JSON structure and keys. Translate every string "
        f"in 'notes' and 'prompts' arrays too (preserve their order and "
        f"count). Do NOT translate names of grammar terms in quotes/parentheses "
        f"that refer to the studied language. Return ONLY the JSON object, no "
        f"commentary.\n\n{json.dumps(payload, ensure_ascii=False)}"
    )


def translate_rule(client, model, mode, base_url, api_key, rule: dict,
                   source_name: str, target_name: str) -> dict | None:
    """Returns a new rule dict with interface fields translated, learning-language
    fields preserved. None on failure."""
    payload = {
        "title": rule.get("title", ""),
        "explanation": rule.get("explanation", ""),
        "notes": [e.get("note", "") for e in rule.get("examples", [])],
        "prompts": [x.get("prompt", "") for x in rule.get("exercises", [])],
    }
    prompt = build_translate_prompt(payload, source_name, target_name)
    for attempt in range(3):
        try:
            out = extract_json(
                _call_model(client, model, mode, prompt, base_url, api_key))
            notes = out.get("notes", []) or []
            prompts = out.get("prompts", []) or []
            examples = []
            for i, e in enumerate(rule.get("examples", [])):
                examples.append({
                    "wrong": e.get("wrong", ""),
                    "right": e.get("right", ""),
                    "note": notes[i] if i < len(notes) else e.get("note", ""),
                })
            exercises = []
            for i, x in enumerate(rule.get("exercises", [])):
                exercises.append({
                    "prompt": prompts[i] if i < len(prompts)
                    else x.get("prompt", ""),
                    "answer": x.get("answer", ""),
                })
            if not out.get("title") or not out.get("explanation"):
                raise ValueError("missing translated title/explanation")
            return {
                "rule_id": rule["rule_id"],
                "topic": rule.get("topic", ""),
                "title": out["title"],
                "explanation": out["explanation"],
                "examples": examples,
                "exercises": exercises,
            }
        except Exception as exc:  # noqa: BLE001 - batch tool, log & retry
            print(f"    ! attempt {attempt + 1} failed: {exc}", file=sys.stderr)
            time.sleep(1.5)
    return None


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass

    import os
    ap = argparse.ArgumentParser(description="Translate a rule set's UI fields.")
    ap.add_argument("--learning", required=True)
    ap.add_argument("--source", default="en", help="interface to translate FROM")
    ap.add_argument("--target", required=True, help="interface to translate TO")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if args.source == args.target:
        return _die("source and target interface are the same")

    api_key = os.getenv("RULES_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        return _die("Set RULES_API_KEY (or DEEPSEEK_API_KEY)")
    base_url = os.getenv("RULES_API_BASE", "https://api.deepseek.com/v1")
    model = os.getenv("RULES_MODEL", "deepseek-v4-flash")
    mode = os.getenv("RULES_API_MODE", "chat").lower()

    src_path = RULES_DIR / f"rules_{args.learning}_{args.source}.json"
    if not src_path.is_file():
        return _die(f"source set not found: {src_path.name}")
    src = json.loads(src_path.read_text("utf-8"))

    out_path = RULES_DIR / f"rules_{args.learning}_{args.target}.json"
    existing = {}
    if out_path.is_file():
        prev = json.loads(out_path.read_text("utf-8"))
        existing = {r["rule_id"]: r for r in prev.get("rules", [])}

    source_name = LANGUAGE_NAMES.get(args.source, args.source)
    target_name = LANGUAGE_NAMES.get(args.target, args.target)
    missing = [r for r in src.get("rules", [])
               if r["rule_id"] not in existing]
    if args.limit > 0:
        missing = missing[: args.limit]

    client = OpenAI(api_key=api_key, base_url=base_url)
    print(f"Translating {args.learning}: {source_name} -> {target_name} "
          f"({len(existing)} existing, {len(missing)} to do, model={model})")

    done = 0
    for i, rule in enumerate(missing, 1):
        print(f"  [{i}/{len(missing)}] {rule['rule_id']}")
        tr = translate_rule(client, model, mode, base_url, api_key, rule,
                            source_name, target_name)
        if not tr:
            print("    -> skipped (translation failed)", file=sys.stderr)
            continue
        existing[rule["rule_id"]] = tr
        done += 1

    # Preserve the source order.
    ordered = [existing[r["rule_id"]] for r in src.get("rules", [])
               if r["rule_id"] in existing]
    out_path.write_text(
        json.dumps({
            "learning": args.learning,
            "interface": args.target,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "translated_from": args.source,
            "rules": ordered,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Translated {done} new; wrote {len(ordered)} rules -> {out_path}")
    return 0


def _die(msg: str) -> int:
    print(f"error: {msg}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
