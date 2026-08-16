"""Tests for the canonical V2 language catalog."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from languages.catalog_v2 import LanguageCatalogError, load_language_catalog


ROOT = Path(__file__).parent


def test_catalog_preserves_all_existing_language_roles():
    catalog = load_language_catalog()
    learning = json.loads((ROOT / "language_configs.json").read_text(encoding="utf-8"))
    interface = json.loads((ROOT / "interface_languages.json").read_text(encoding="utf-8"))
    assert {language.code for language in catalog.languages if "learning" in language.roles} == {
        code.replace("_", "-") for code in learning
    }
    assert {language.code for language in catalog.languages if "interface" in language.roles} == {
        code.replace("_", "-") for code in interface
    }


def test_legacy_locale_alias_and_detector_are_data_driven():
    catalog = load_language_catalog()
    spanish_mexico = catalog.language("es_MX")
    assert spanish_mexico.code == "es-MX"
    assert spanish_mexico.detection_code == "es"
    assert catalog.language("AR").direction == "rtl"


def test_level_descriptions_cover_every_configured_interface_language():
    catalog = load_language_catalog()
    assert catalog.level_description("B2", "ru") != catalog.level_description("B2", "en")
    assert catalog.level_description("B2", "el") != catalog.level_description("B2", "en")

    for level in ("A1", "A2", "B1", "B2", "C1", "C2"):
        for language in catalog.languages:
            if "interface" in language.roles:
                assert catalog.level_description(level, language.code).strip()


def test_compatibility_views_are_derived_from_v2_without_parallel_data():
    from language_catalog import load_catalog, runtime_views

    learning, interfaces, levels, contexts = runtime_views(load_catalog())
    assert learning["es_MX"]["code"] == "es_MX"
    assert interfaces["es_MX"]["name"] == "Español Mexicano"
    assert contexts["es_MX"]["no_context"]
    assert levels["A1"]["description"]["ru"]


def test_duplicate_alias_is_rejected():
    data = json.loads((ROOT / "languages" / "catalog.v2.json").read_text(encoding="utf-8"))
    data["languages"]["en"]["legacy_codes"] = ["shared"]
    data["languages"]["de"]["legacy_codes"] = ["shared"]
    with TemporaryDirectory() as directory:
        path = Path(directory) / "catalog.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        try:
            load_language_catalog(path)
            assert False, "expected duplicate alias"
        except LanguageCatalogError as error:
            assert "maps to both" in str(error)


def test_non_interface_language_cannot_supply_level_description():
    data = json.loads((ROOT / "languages" / "catalog.v2.json").read_text(encoding="utf-8"))
    data["level_profiles"]["A1"]["descriptions"]["xx"] = "not allowed"
    with TemporaryDirectory() as directory:
        path = Path(directory) / "catalog.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        try:
            load_language_catalog(path)
            assert False, "expected invalid level description language"
        except LanguageCatalogError as error:
            assert "not a configured interface" in str(error)


def test_missing_interface_level_description_is_rejected():
    data = json.loads((ROOT / "languages" / "catalog.v2.json").read_text(encoding="utf-8"))
    del data["level_profiles"]["A1"]["descriptions"]["el"]
    with TemporaryDirectory() as directory:
        path = Path(directory) / "catalog.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        try:
            load_language_catalog(path)
        except LanguageCatalogError as error:
            assert "missing interface descriptions" in str(error)
        else:
            raise AssertionError("expected LanguageCatalogError")


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
