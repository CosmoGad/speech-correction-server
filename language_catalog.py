"""Compatibility views over the validated V2 language catalog.

New code should use :mod:`languages.catalog_v2` directly. These views let the
released V1 endpoint consume the same single source while legacy clients still
send locale identifiers such as ``es_MX``.
"""

from __future__ import annotations

import json
from pathlib import Path

from languages.catalog_v2 import LanguageCatalogError, load_language_catalog


ROOT = Path(__file__).parent
DEFAULT_PATH = ROOT / "languages" / "catalog.v2.json"


class CatalogError(ValueError):
    """The server cannot start with an invalid language catalog."""


def load_catalog(path: Path = DEFAULT_PATH) -> dict:
    """Load raw data only after the strict V2 validator accepts it."""
    try:
        load_language_catalog(path)
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, LanguageCatalogError) as error:
        raise CatalogError(f"Unable to load language catalog: {error}") from error


def runtime_views(catalog: dict) -> tuple[dict, dict, dict, dict]:
    """Expose legacy-keyed views without a parallel configuration source."""
    source_languages = catalog["languages"]
    learning: dict = {}
    interfaces: dict = {}
    contexts: dict = {}
    canonical_to_runtime: dict[str, str] = {}
    for canonical, item in source_languages.items():
        runtime_code = item.get("legacy_codes", [canonical])[0]
        canonical_to_runtime[canonical] = runtime_code
        if "learning" in item["roles"]:
            learning[runtime_code] = {
                "code": runtime_code,
                "common_errors": item["common_errors"],
                "pronunciation_focus": item["pronunciation_focus"],
            }
            contexts[runtime_code] = item["context"]
        if "interface" in item["roles"]:
            interfaces[runtime_code] = {
                "name": item["native_name"],
                "language_code": runtime_code,
            }

    levels: dict = {}
    for level, profile in catalog["level_profiles"].items():
        converted = {key: value for key, value in profile.items() if key != "descriptions"}
        converted["description"] = {
            canonical_to_runtime[code]: value
            for code, value in profile["descriptions"].items()
        }
        levels[level] = converted
    return learning, interfaces, levels, contexts


def display_name(catalog: dict, code: str) -> str:
    """Return the configured native name for a canonical or legacy code."""
    normalized = code.replace("_", "-")
    for canonical, item in catalog.get("languages", {}).items():
        aliases = item.get("legacy_codes", [])
        if normalized == canonical or code in aliases:
            return item["native_name"]
    return code
