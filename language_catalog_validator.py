"""Read-only audit for language catalog v1 and the target v2 contract.

This is intentionally not imported by the production server: it exposes
inconsistencies while the catalog migration is still behind a feature flag.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class CatalogIssue:
    code: str
    message: str
    path: str


def _read_json(root: Path, filename: str) -> dict[str, Any]:
    with (root / filename).open(encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise ValueError(f"{filename} must contain a JSON object")
    return value


def _duplicate_values(values: dict[str, Any]) -> set[str]:
    counts = Counter(value for value in values.values() if isinstance(value, str) and value)
    return {value for value, count in counts.items() if count > 1}


def validate_catalog(root: Path | str = ROOT) -> list[CatalogIssue]:
    """Return cross-file configuration problems without mutating any file."""
    root = Path(root)
    language_configs = _read_json(root, "language_configs.json")
    interface_languages = _read_json(root, "interface_languages.json")
    level_details = _read_json(root, "level_details.json")
    context_instructions = _read_json(root, "context_instructions.json")
    issues: list[CatalogIssue] = []

    learning_codes, interface_codes = set(language_configs), set(interface_languages)
    context_codes = set(context_instructions)
    prompt_paths = sorted((root / "prompts").glob("prompt_*.json"))
    prompt_codes = {path.stem.removeprefix("prompt_") for path in prompt_paths}
    for code, config in language_configs.items():
        if not isinstance(config, dict) or config.get("code") != code:
            issues.append(CatalogIssue("LEARNING_LANGUAGE_CODE_MISMATCH", f"language_configs[{code!r}].code must equal {code!r}", "language_configs.json"))
        if not isinstance(config, dict) or not config.get("detection_code"):
            issues.append(CatalogIssue("MISSING_DETECTION_CODE", f"Learning language {code!r} has no detection_code.", "language_configs.json"))

    identities = {key: value.get("language_code") if isinstance(value, dict) else None for key, value in interface_languages.items()}
    for duplicate in _duplicate_values(identities):
        issues.append(CatalogIssue("DUPLICATE_INTERFACE_LANGUAGE_CODE", f"More than one interface entry uses language_code {duplicate!r}.", "interface_languages.json"))
    for code, identity in identities.items():
        if identity != code:
            issues.append(CatalogIssue("INTERFACE_LANGUAGE_CODE_MISMATCH", f"interface_languages[{code!r}].language_code must equal {code!r}", "interface_languages.json"))

    for code in sorted(learning_codes - prompt_codes):
        issues.append(CatalogIssue("MISSING_PROMPT_FILE", f"No prompts/prompt_{code}.json exists.", "prompts"))
    for code in sorted(prompt_codes - learning_codes):
        issues.append(CatalogIssue("PROMPT_WITHOUT_LEARNING_CONFIG", f"prompt_{code}.json has no learning-language configuration.", "prompts"))
    for code in sorted(learning_codes - context_codes):
        issues.append(CatalogIssue("MISSING_CONTEXT_INSTRUCTION", f"Learning language {code!r} has no context instruction.", "context_instructions.json"))
    for code in sorted(context_codes - learning_codes):
        issues.append(CatalogIssue("CONTEXT_WITHOUT_LEARNING_CONFIG", f"Context instruction {code!r} has no learning-language configuration.", "context_instructions.json"))

    known_codes = interface_codes | learning_codes
    for level, detail in level_details.items():
        descriptions = detail.get("description", {}) if isinstance(detail, dict) else {}
        if not isinstance(descriptions, dict):
            issues.append(CatalogIssue("INVALID_LEVEL_DESCRIPTION", f"Level {level!r} has no description object.", "level_details.json"))
            continue
        for code in descriptions:
            if code not in known_codes:
                issues.append(CatalogIssue("LEVEL_DESCRIPTION_NON_ISO_KEY", f"Level {level!r} uses {code!r}; descriptions must use configured ISO codes.", "level_details.json"))

    hashes: dict[str, str] = {}
    for path in prompt_paths:
        try:
            payload = _read_json(path.parent, path.name)
            prompt = payload.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError("Prompt must be a non-empty string.")
            hashes[path.name] = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        except (OSError, ValueError, json.JSONDecodeError) as error:
            issues.append(CatalogIssue("INVALID_PROMPT_FILE", str(error), str(path.relative_to(root))))
    for digest in set(hashes.values()):
        duplicates = sorted(name for name, value in hashes.items() if value == digest)
        if len(duplicates) > 1:
            issues.append(CatalogIssue("DUPLICATE_PROMPT_CONTENT", f"Prompt content is duplicated in: {', '.join(duplicates)}.", "prompts"))
    return issues


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Audit language catalog configuration.")
    parser.add_argument("--root", type=Path, default=ROOT)
    arguments = parser.parse_args()
    issues = validate_catalog(arguments.root)
    for issue in issues:
        print(f"{issue.code}: {issue.path}: {issue.message}")
    if issues:
        print(f"Found {len(issues)} issue(s).")
        return 1
    print("Language catalog validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
