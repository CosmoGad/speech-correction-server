"""Tests for rules_store (pure data access, no FastAPI needed).

Run with pytest, or directly: `python test_rules_store.py`.
Requires the generated rules/rules_en_ru.json to be present.
"""

import rules_store as s


def test_list_rules_shape():
    lst = s.list_rules("en", "ru")
    assert len(lst) > 0
    assert set(lst[0].keys()) == {"rule_id", "title"}


def test_get_rule_full_content():
    r = s.get_rule("en", "ru", "articles")
    assert r["rule_id"] == "articles"
    assert "explanation" in r and "examples" in r and "exercises" in r


def test_rule_not_found():
    try:
        s.get_rule("en", "ru", "definitely-not-a-rule")
        assert False, "expected RuleNotFound"
    except s.RuleNotFound:
        pass


def test_rules_not_found():
    try:
        s.list_rules("en", "zz")
        assert False, "expected RulesNotFound"
    except s.RulesNotFound:
        pass


def test_path_traversal_blocked():
    for bad in ("../../etc/passwd", "a/b", ".."):
        try:
            s.list_rules(bad, "ru")
            assert False, f"expected ValueError for {bad!r}"
        except ValueError:
            pass


def test_bad_rule_id_rejected():
    try:
        s.get_rule("en", "ru", "../secret")
        assert False, "expected ValueError"
    except ValueError:
        pass


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
