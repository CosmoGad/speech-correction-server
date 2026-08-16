"""Validated single source of truth for backend language metadata.

The V2 catalog uses canonical BCP-47-like codes while retaining declared
legacy aliases (such as ``es_MX``) during client migration.  It has no
language-specific conditionals: adding a language is a data change.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


class LanguageCatalogError(ValueError):
    """Raised when a catalog is malformed or unsafe to activate."""


@dataclass(frozen=True)
class Language:
    code: str
    native_name: str
    detection_code: str
    direction: str
    roles: frozenset[str]
    interface_labels: dict[str, str]


class LanguageCatalog:
    def __init__(self, languages: dict[str, Language], aliases: dict[str, str], levels: dict[str, dict[str, Any]], default_interface: str) -> None:
        self._languages = languages
        self._aliases = aliases
        self._levels = levels
        self.default_interface = default_interface

    @property
    def languages(self) -> tuple[Language, ...]:
        return tuple(self._languages.values())

    def canonical_code(self, code: str) -> str:
        normalized = _normalize_code(code)
        canonical = self._aliases.get(normalized, normalized)
        if canonical not in self._languages:
            raise LanguageCatalogError(f"unsupported language code: {code}")
        return canonical

    def language(self, code: str) -> Language:
        return self._languages[self.canonical_code(code)]

    def level_description(self, level: str, interface_code: str) -> str:
        language = self.language(interface_code)
        if "interface" not in language.roles:
            raise LanguageCatalogError(f"language is not an interface language: {interface_code}")
        profile = self._levels.get(level.upper())
        if profile is None:
            raise LanguageCatalogError(f"unsupported CEFR level: {level}")
        return profile["descriptions"][language.code]


def load_language_catalog(path: str | Path | None = None) -> LanguageCatalog:
    catalog_path = Path(path) if path else Path(__file__).with_name("catalog.v2.json")
    try:
        raw = json.loads(catalog_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise LanguageCatalogError(f"catalog file is missing: {catalog_path}") from error
    except json.JSONDecodeError as error:
        raise LanguageCatalogError(f"catalog is not valid JSON: {error.msg}") from error
    if not isinstance(raw, dict) or raw.get("schema_version") != 2:
        raise LanguageCatalogError("catalog.schema_version must be 2")
    source_languages = raw.get("languages")
    source_levels = raw.get("level_profiles")
    if not isinstance(source_languages, dict) or not isinstance(source_levels, dict):
        raise LanguageCatalogError("catalog must define languages and level_profiles objects")

    languages: dict[str, Language] = {}
    aliases: dict[str, str] = {}
    for key, raw_language in source_languages.items():
        language, language_aliases = _parse_language(key, raw_language)
        if language.code in languages:
            raise LanguageCatalogError(f"duplicate canonical language code: {language.code}")
        languages[language.code] = language
        for alias in {language.code, *language_aliases}:
            previous = aliases.setdefault(alias, language.code)
            if previous != language.code:
                raise LanguageCatalogError(f"alias {alias!r} maps to both {previous!r} and {language.code!r}")

    default_interface = _normalize_code(raw.get("default_interface_language", ""))
    if default_interface not in languages or "interface" not in languages[default_interface].roles:
        raise LanguageCatalogError("default_interface_language must be a configured interface language")
    levels = _parse_levels(source_levels, languages, default_interface)
    return LanguageCatalog(languages, aliases, levels, default_interface)


def _parse_language(key: Any, raw: Any) -> tuple[Language, set[str]]:
    if not isinstance(key, str) or not isinstance(raw, dict):
        raise LanguageCatalogError("each language entry must be an object keyed by its code")
    code = _normalize_code(raw.get("code", ""))
    if code != _normalize_code(key):
        raise LanguageCatalogError(f"language key {key!r} does not match entry code {raw.get('code')!r}")
    name = raw.get("native_name")
    detection_code = raw.get("detection_code")
    direction = raw.get("direction")
    roles = raw.get("roles")
    if not isinstance(name, str) or not name.strip():
        raise LanguageCatalogError(f"{code}: native_name is required")
    if not isinstance(detection_code, str) or not detection_code.strip():
        raise LanguageCatalogError(f"{code}: detection_code is required")
    if direction not in {"ltr", "rtl"}:
        raise LanguageCatalogError(f"{code}: direction must be ltr or rtl")
    if not isinstance(roles, list) or not roles or set(roles) - {"learning", "interface"}:
        raise LanguageCatalogError(f"{code}: roles must contain learning and/or interface")
    labels = raw.get("interface_labels", {})
    if "interface" in roles:
        if not isinstance(labels, dict) or not all(isinstance(k, str) and isinstance(v, str) and v.strip() for k, v in labels.items()):
            raise LanguageCatalogError(f"{code}: interface_labels must be a non-empty string map")
    elif labels:
        raise LanguageCatalogError(f"{code}: non-interface language cannot define interface_labels")
    aliases = {_normalize_code(alias) for alias in raw.get("legacy_codes", [])}
    return Language(code, name, detection_code, direction, frozenset(roles), dict(labels)), aliases


def _parse_levels(raw_levels: dict[str, Any], languages: dict[str, Language], default_interface: str) -> dict[str, dict[str, Any]]:
    levels: dict[str, dict[str, Any]] = {}
    for level, raw_profile in raw_levels.items():
        if not isinstance(level, str) or not isinstance(raw_profile, dict):
            raise LanguageCatalogError("level_profiles must map levels to objects")
        descriptions = raw_profile.get("descriptions")
        if not isinstance(descriptions, dict) or not descriptions:
            raise LanguageCatalogError(f"{level}: descriptions must be a non-empty object")
        normalized_descriptions: dict[str, str] = {}
        for code, value in descriptions.items():
            canonical = _normalize_code(code)
            language = languages.get(canonical)
            if language is None or "interface" not in language.roles:
                raise LanguageCatalogError(f"{level}: description language is not a configured interface: {code}")
            if not isinstance(value, str) or not value.strip():
                raise LanguageCatalogError(f"{level}: description for {code} is empty")
            normalized_descriptions[canonical] = value
        if default_interface not in normalized_descriptions:
            raise LanguageCatalogError(f"{level}: missing default interface description")
        missing_interfaces = sorted(
            code for code, language in languages.items()
            if "interface" in language.roles and code not in normalized_descriptions
        )
        if missing_interfaces:
            raise LanguageCatalogError(
                f"{level}: missing interface descriptions for "
                + ", ".join(missing_interfaces)
            )
        profile = dict(raw_profile)
        profile["descriptions"] = normalized_descriptions
        levels[level.upper()] = profile
    return levels


def _normalize_code(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LanguageCatalogError("language code must be a non-empty string")
    parts = value.strip().replace("_", "-").split("-")
    if any(not part.isalnum() for part in parts):
        raise LanguageCatalogError(f"invalid language code: {value!r}")
    return "-".join([parts[0].lower(), *[part.upper() if len(part) == 2 and part.isalpha() else part for part in parts[1:]]])
