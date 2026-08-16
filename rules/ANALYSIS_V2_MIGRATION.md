# Analysis V2 migration contract

## Goal

Make correction, rule mapping, and learning status data-driven. Adding a
language or a rule must not require a language-specific conditional in Python
or Flutter.

## Non-negotiable rules

1. A system prompt contains instructions only. User text and intent/context are
   passed in a separate structured user message.
2. A text-only request never claims to evaluate pronunciation. Audio analysis is
   a separate capability.
3. Error categories are a closed contract:
   `grammar`, `vocabulary`, `spelling`, `word_order`, `style`, `other`.
4. Rule selection is deterministic after analysis. It uses declared
   `concept_codes`, never language keywords or a "closest rule" fallback.
5. If no mapping exists, return `unresolved`; do not silently choose a rule.
6. A language is configured by data: ISO code, display names, detector aliases,
   direction, and supported capabilities.

## Analysis response shape

```json
{
  "contract_version": 2,
  "corrected_text": "...",
  "errors": [
    {
      "type": "grammar",
      "concept_code": "verb.conjugation.present",
      "confidence": 0.94,
      "original": "...",
      "corrected": "...",
      "explanation": "..."
    }
  ],
  "explanation": "...",
  "grammar_notes": "...",
  "level_suggestions": "...",
  "alternatives": [{"sentence": "...", "explanation": "..."}]
}
```

`original` must be an exact substring of the submitted text and `corrected`
must be an exact substring of `corrected_text` when the request contains text.
Intent-only requests may have no errors.

## Rule mapping shape

The current taxonomy derives one stable opaque `concept_code` from each active
`(learning, rule_id)` pair. The code is sent beside the human title and maps
back exactly without a second model call. A future explicitly curated registry
is permitted only after its entries pass duplicate/missing/inactive validation;
it must never become a keyword fallback.

```json
{
  "rule_id": "subject-verb-agreement",
  "concept_code": "taxonomy.4b20e5d77db1b7b7f824"
}
```

The server maps each `concept_code` to at most one active topic for a learning
language. Invalid or absent mappings become `unresolved` and are logged with
aggregate counters only.

## Migration and rollback

1. Introduce V2 modules and validate them with the committed eval set.
2. Run V1 and V2 in shadow mode; do not show V2 to users yet.
3. Compare validity, language correctness, correction coverage, mapping rate,
   p95 latency, and token use.
4. Enable V2 behind a server feature flag for a small cohort.
5. Remove V1 prompts, `fallback_rule_id`, and V1 resolver cache only after the
   V2 acceptance gate passes.

Rollback is a feature-flag change. It must never re-enable a language-specific
fallback.

## Acceptance gates

- All configured languages pass catalogue validation.
- Every prompt eval case produces schema-valid output or a typed failure.
- No production path contains `if learning == ...` or keyword-to-rule mapping.
- A same-language interface/learning pair has no conflicting language rule.
- No user text or intent appears in the system prompt.
- Unknown concepts never open an unrelated rule.
