import sqlite3
import json
import os
import logging

logger = logging.getLogger(__name__)

def import_rules_from_json(language, json_file):
    """Импортирует правила из JSON-файла в базу данных."""
    with open(json_file, 'r', encoding='utf-8') as f:
        rules = json.load(f)

    conn = sqlite3.connect("grammar_codex.db")
    c = conn.cursor()

    for rule in rules:
        # Предполагаем, что уровень по умолчанию — A1, если не указан
        level = rule.get('level', 'A1')
        # Нормализуем имя правила (убираем пробелы, приводим к нижнему регистру)
        normalized_rule_name = rule['rule_name'].replace(' ', '').lower()
        examples_json = json.dumps(rule['examples'], ensure_ascii=False)

        c.execute("""
            INSERT OR IGNORE INTO GrammarRules (language, level, rule_name, description, examples, normalized_rule_name)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            language,
            level,
            rule['rule_name'],
            rule['description'],
            examples_json,
            normalized_rule_name
        ))

    conn.commit()
    conn.close()

def init_db():
    conn = sqlite3.connect("grammar_codex.db")
    c = conn.cursor()

    # Создание таблиц (без изменений)
    c.execute("""
        CREATE TABLE IF NOT EXISTS GrammarRules (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            language TEXT,
            level TEXT,
            rule_name TEXT,
            description TEXT,
            examples TEXT,
            normalized_rule_name TEXT UNIQUE
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS MiniTests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            rule_id INTEGER,
            type TEXT,
            question_template TEXT,
            options TEXT,
            correct_answer TEXT,
            explanation_template TEXT,
            FOREIGN KEY (rule_id) REFERENCES GrammarRules(id)
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS UserProgress (
            user_id TEXT,
            rule_id INTEGER,
            status TEXT,
            last_test_score REAL,
            last_test_date TEXT,
            FOREIGN KEY (rule_id) REFERENCES GrammarRules(id)
        )
    """)

    # Импорт правил для русского языка
    ru_rules_file = "grammar_rules_ru.json"
    if os.path.exists(ru_rules_file):
        import_rules_from_json("ru", ru_rules_file)
        logger.info(f"Imported rules for Russian from {ru_rules_file}")

    # Заглушки для других языков (можно добавить файлы позже)
    other_languages = ['en', 'de', 'uk', 'it', 'fr', 'es', 'pl', 'ar', 'da', 'fa', 'nl', 'no', 'pt', 'sv', 'tr', 'ur', 'sr', 'es_MX']
    for lang in other_languages:
        lang_rules_file = f"grammar_rules_{lang}.json"
        if os.path.exists(lang_rules_file):
            import_rules_from_json(lang, lang_rules_file)
            logger.info(f"Imported rules for {lang} from {lang_rules_file}")
        else:
            # Добавляем одно тестовое правило, чтобы язык был в базе
            c.execute("""
                INSERT OR IGNORE INTO GrammarRules (language, level, rule_name, description, examples, normalized_rule_name)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                lang,
                "A1",
                f"Basic rule for {lang}",
                f"Placeholder rule for {lang}.",
                json.dumps([]),
                f"basicrule_{lang}"
            ))

    conn.commit()
    conn.close()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    init_db()
    logger.info("База данных grammar_codex.db создана с начальными данными.")
