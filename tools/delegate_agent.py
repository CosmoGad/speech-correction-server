#!/usr/bin/env python3
"""Minimal unattended coding agent powered by a cheap model (DeepSeek/Kimi/MiMo
via the OpenModel gateway). Experimental "delegation lever":

    Claude writes a spec  ->  this runner makes the model do the work inside an
    isolated directory (a git worktree)  ->  Claude reviews the diff & merges.

Design notes:
- Provider-agnostic, reuses the same env as generate_rules.py:
      RULES_API_KEY / RULES_API_BASE / RULES_MODEL / RULES_API_MODE
  Defaults target OpenModel's DeepSeek (Anthropic Messages protocol).
- Tools are exposed via a *prompt protocol* (the model replies with ONE JSON
  action per turn), so it works on any chat/messages model without relying on
  native function-calling.
- SAFETY: every file operation is sandboxed to --dir; paths escaping it are
  rejected. Shell `run` is disabled unless --allow-run, and then limited to an
  allowlist of test/analyze commands.

Usage:
    python tools/delegate_agent.py --dir <worktree> --spec-file spec.md
    python tools/delegate_agent.py --dir . --spec "Add a /health route" --max-steps 8
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

try:
    import httpx
except ImportError:
    sys.exit("The 'httpx' package is required: pip install httpx")

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# Directories we never show or let the agent traverse.
SKIP_DIRS = {".git", "build", "node_modules", ".dart_tool", "__pycache__",
             "translate_env", ".venv", "venv", ".idea", ".gradle"}
# Only these command prefixes are allowed when --allow-run is set.
RUN_ALLOWLIST = ("flutter test", "flutter analyze", "dart ", "python -m pytest",
                 "pytest", "python -m py_compile")

SYSTEM_PROMPT = """You are an autonomous coding agent working inside a single \
sandboxed directory. You CANNOT see or touch anything outside it.

You MUST reply with EXACTLY ONE JSON object and nothing else (no prose, no \
markdown fences). Allowed actions:

  {"action":"list_dir","path":"."}                      list files (recursive)
  {"action":"read_file","path":"rel/path"}              read a file
  {"action":"write_file","path":"rel/path","content":"..."}  create/overwrite
  {"action":"run","cmd":"flutter test"}                 run an allowed command
  {"action":"done","summary":"what you changed"}        finish

Rules:
- Paths are relative to the sandbox root. Never use absolute paths or "..".
- Make the smallest change that satisfies the spec. Match the existing style.
- Read before you write. Verify by running tests if running is available.
- When the spec is fully satisfied, reply with the "done" action.
- One action per turn. Wait for the tool result before the next action."""


def build_client_cfg():
    key = os.getenv("RULES_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    if not key:
        sys.exit("Set RULES_API_KEY (or DEEPSEEK_API_KEY) in env/.env")
    return {
        "key": key,
        "base": os.getenv("RULES_API_BASE", "https://api.deepseek.com/v1"),
        "model": os.getenv("RULES_MODEL", "deepseek-v4-flash"),
        "mode": os.getenv("RULES_API_MODE", "chat").lower(),
    }


def call_model(cfg: dict, messages: list[dict]) -> str:
    """One assistant turn. Supports Anthropic Messages and OpenAI Chat."""
    base = cfg["base"].rstrip("/")
    headers = {"Authorization": f"Bearer {cfg['key']}",
               "Content-Type": "application/json"}
    if cfg["mode"] == "messages":
        body = {"model": cfg["model"], "max_tokens": 4000, "temperature": 0.2,
                "system": SYSTEM_PROMPT, "messages": messages}
        r = httpx.post(f"{base}/messages", headers=headers, json=body, timeout=180)
        r.raise_for_status()
        payload = r.json()
        if isinstance(payload.get("data"), dict):
            payload = payload["data"]
        return "\n".join(b.get("text", "") for b in payload.get("content", [])
                         if b.get("type") == "text").strip()
    # OpenAI chat completions
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
    body = {"model": cfg["model"], "max_tokens": 4000, "temperature": 0.2,
            "messages": msgs}
    r = httpx.post(f"{base}/chat/completions", headers=headers, json=body,
                   timeout=180)
    r.raise_for_status()
    return (r.json()["choices"][0]["message"]["content"] or "").strip()


def parse_action(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        s, e = text.find("{"), text.rfind("}")
        if s != -1 and e > s:
            return json.loads(text[s:e + 1])
        raise


def safe_path(base: Path, rel: str) -> Path:
    target = (base / rel).resolve()
    if base not in target.parents and target != base:
        raise ValueError(f"path escapes sandbox: {rel}")
    return target


def tool_list_dir(base: Path, rel: str) -> str:
    root = safe_path(base, rel or ".")
    out = []
    for p in sorted(root.rglob("*")):
        if any(part in SKIP_DIRS for part in p.relative_to(base).parts):
            continue
        if p.is_file():
            out.append(str(p.relative_to(base)).replace("\\", "/"))
    return "\n".join(out[:500]) or "(empty)"


def tool_read_file(base: Path, rel: str) -> str:
    p = safe_path(base, rel)
    if not p.is_file():
        return f"ERROR: not a file: {rel}"
    text = p.read_text(encoding="utf-8", errors="replace")
    return text[:20000]


def tool_write_file(base: Path, rel: str, content: str) -> str:
    p = safe_path(base, rel)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"OK: wrote {len(content)} chars to {rel}"


def tool_run(base: Path, cmd: str, allow_run: bool) -> str:
    if not allow_run:
        return "ERROR: running commands is disabled (start with --allow-run)"
    if not any(cmd.strip().startswith(p) for p in RUN_ALLOWLIST):
        return f"ERROR: command not allowed. Allowed: {RUN_ALLOWLIST}"
    try:
        proc = subprocess.run(cmd, shell=True, cwd=str(base), timeout=600,
                              capture_output=True, text=True)
        tail = (proc.stdout + proc.stderr)[-4000:]
        return f"exit={proc.returncode}\n{tail}"
    except subprocess.TimeoutExpired:
        return "ERROR: command timed out"


def main() -> int:
    ap = argparse.ArgumentParser(description="Unattended delegation agent.")
    ap.add_argument("--dir", required=True, help="sandbox directory (worktree)")
    ap.add_argument("--spec", help="task spec text")
    ap.add_argument("--spec-file", help="task spec file")
    ap.add_argument("--max-steps", type=int, default=12)
    ap.add_argument("--allow-run", action="store_true",
                    help="permit a small allowlist of test/analyze commands")
    args = ap.parse_args()

    base = Path(args.dir).resolve()
    if not base.is_dir():
        sys.exit(f"--dir is not a directory: {base}")
    if args.spec_file:
        spec = Path(args.spec_file).read_text(encoding="utf-8")
    elif args.spec:
        spec = args.spec
    else:
        sys.exit("Provide --spec or --spec-file")

    cfg = build_client_cfg()
    print(f"agent: model={cfg['model']} mode={cfg['mode']} dir={base} "
          f"allow_run={args.allow_run}\n")

    tree = tool_list_dir(base, ".")
    first = (f"SPEC:\n{spec}\n\nSANDBOX FILE TREE:\n{tree}\n\n"
             "Begin. Reply with one JSON action.")
    messages = [{"role": "user", "content": first}]

    for step in range(1, args.max_steps + 1):
        try:
            reply = call_model(cfg, messages)
            action = parse_action(reply)
        except Exception as exc:  # noqa: BLE001
            print(f"[{step}] model/parse error: {exc}", file=sys.stderr)
            return 1
        messages.append({"role": "assistant", "content": reply})
        act = action.get("action")
        print(f"[{step}] {act} {action.get('path', action.get('cmd', ''))}")

        if act == "done":
            print(f"\nDONE: {action.get('summary', '')}")
            return 0
        try:
            if act == "list_dir":
                result = tool_list_dir(base, action.get("path", "."))
            elif act == "read_file":
                result = tool_read_file(base, action["path"])
            elif act == "write_file":
                result = tool_write_file(base, action["path"],
                                         action.get("content", ""))
            elif act == "run":
                result = tool_run(base, action.get("cmd", ""), args.allow_run)
            else:
                result = f"ERROR: unknown action '{act}'"
        except Exception as exc:  # noqa: BLE001
            result = f"ERROR: {exc}"
        messages.append({"role": "user", "content": f"TOOL RESULT:\n{result}"})

    print("\nstopped: reached --max-steps without 'done'", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
