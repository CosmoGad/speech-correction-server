import json
import tempfile
import unittest
from pathlib import Path

from language_catalog_validator import validate_catalog


def write_json(root, name, value):
    (root / name).write_text(json.dumps(value), encoding="utf-8")


def valid_catalog(root):
    write_json(root, "language_configs.json", {"en": {"code": "en", "detection_code": "en"}})
    write_json(root, "interface_languages.json", {"en": {"language_code": "en"}})
    write_json(root, "level_details.json", {"A1": {"description": {"en": "Beginner"}}})
    write_json(root, "context_instructions.json", {"en": {"with_context": "{context}", "no_context": "None"}})
    prompts = root / "prompts"; prompts.mkdir()
    (prompts / "prompt_en.json").write_text(json.dumps({"prompt": "Unique prompt"}), encoding="utf-8")


class LanguageCatalogValidatorTest(unittest.TestCase):
    def catalog(self):
        directory = tempfile.TemporaryDirectory(); self.addCleanup(directory.cleanup)
        root = Path(directory.name); valid_catalog(root); return root

    def test_valid_catalog_has_no_issues(self):
        self.assertEqual(validate_catalog(self.catalog()), [])

    def test_reports_missing_detection_code_and_non_iso_level_key(self):
        root = self.catalog()
        write_json(root, "language_configs.json", {"en": {"code": "en"}})
        write_json(root, "level_details.json", {"A1": {"description": {"English": "Beginner"}}})
        codes = {issue.code for issue in validate_catalog(root)}
        self.assertIn("MISSING_DETECTION_CODE", codes)
        self.assertIn("LEVEL_DESCRIPTION_NON_ISO_KEY", codes)

    def test_reports_set_mismatch_and_copied_prompts(self):
        root = self.catalog()
        write_json(root, "language_configs.json", {"en": {"code": "en", "detection_code": "en"}, "de": {"code": "de", "detection_code": "de"}})
        (root / "prompts" / "prompt_de.json").write_text(json.dumps({"prompt": "Unique prompt"}), encoding="utf-8")
        codes = {issue.code for issue in validate_catalog(root)}
        self.assertIn("MISSING_CONTEXT_INSTRUCTION", codes)
        self.assertIn("DUPLICATE_PROMPT_CONTENT", codes)


if __name__ == "__main__":
    unittest.main()
