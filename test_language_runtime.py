"""Runtime regressions for generic language/locale compatibility."""

import speech_correction_server as server


def test_legacy_level_descriptions_use_interface_name_without_manual_language_map():
    level = {"description": {"English": "Beginner", "Русский": "Начальный уровень"}}
    interface = {"language_code": "ru", "name": "Русский"}
    assert server._level_description(level, interface) == "Начальный уровень"


def test_iso_level_description_wins_over_legacy_display_name():
    level = {"description": {"ru": "ISO Russian", "Русский": "Legacy Russian"}}
    interface = {"language_code": "ru", "name": "Русский"}
    assert server._level_description(level, interface) == "ISO Russian"


def test_language_detection_normalizes_locales_to_detector_base_code():
    original_detect = server.detect_langs

    class Guess:
        lang = "es"
        prob = 0.99

    try:
        server.detect_langs = lambda _: [Guess()]
        assert not server._is_wrong_language("Esta frase tiene suficiente longitud para detectar idioma.", "es_MX")
    finally:
        server.detect_langs = original_detect


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
