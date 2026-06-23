# Rule book (grammar rules collection)

Pre-generated, free-to-serve grammar lessons keyed to the common mistakes of a
learning language. Goal: after a correction, the user can tap "Learn more about
this rule" and read a short lesson + try a couple of exercises — without
spending model tokens at runtime.

## Strategy (combined)

1. **Batch (this folder).** Pre-generate lessons for the most common topics
   (`common_errors` + `pronunciation_focus` from `language_configs.json`) using
   `tools/generate_rules.py`. Generation can run against a **free** endpoint to
   save tokens (see below). Output is committed static JSON — production serves
   it with zero model calls.
2. **On-demand (later).** For a rule that is not pre-generated, the server
   generates it once and stores it in the existing `ResponseCache` (SQLite),
   same content-addressed pattern as `/process-text/`.

## File layout

One file per (learning, interface) pair:

    rules/rules_<learning>_<interface>.json

A rule explains a topic of the *learning* language, written in the *interface*
(user's) language. We pre-generate only the popular pairs (e.g. interface
`en`/`ru` × top learning languages); everything else falls back to on-demand.

## Schema

```jsonc
{
  "learning": "en",
  "interface": "ru",
  "generated_at": "2026-06-21T...Z",
  "model": "deepseek-v4-flash",
  "rules": [
    {
      "rule_id": "articles",          // slug of the topic, stable id
      "topic": "Articles",            // seed topic (learning language)
      "title": "...",                 // interface language
      "explanation": "...",           // interface language, 2-4 sentences
      "examples": [
        { "wrong": "...",             // learning language
          "right": "...",             // learning language
          "note": "..." }             // interface language
      ],
      "exercises": [
        { "prompt": "...",            // interface language
          "answer": "..." }           // learning language
      ]
    }
  ]
}
```

## Generating

```bash
# Provider is configurable; point it at a free endpoint to save tokens.
#   RULES_API_KEY   (falls back to DEEPSEEK_API_KEY)
#   RULES_API_BASE  (default https://api.deepseek.com/v1)
#   RULES_MODEL     (default deepseek-v4-flash)
#   RULES_API_MODE  chat | responses | messages (default chat)

python tools/generate_rules.py --learning en --interface en           # full set
python tools/generate_rules.py --learning el --interface ru --limit 3 # sample
```

### Free generation via OpenModel (proven 2026-06-21)

OpenModel serves DeepSeek through the **Anthropic Messages protocol**, not Chat
Completions. Put this in the server `.env`:

```
RULES_API_KEY=<your OpenModel key>
RULES_API_BASE=https://api.openmodel.ai/v1
RULES_MODEL=deepseek-v4-flash      # or deepseek-v4-pro
RULES_API_MODE=messages
```

(The OpenAI Chat Completions / Responses routes 404 on OpenModel for DeepSeek;
`GET /v1/models` lists the available ids. The free DeepSeek event runs to
2026-06-28 — use it for batch generation only, not as a production runtime.)

## Server endpoint (sketch — not yet wired)

`GET /rule?learning=en&interface=ru&rule_id=articles` →

1. Look in `rules/rules_<learning>_<interface>.json` for `rule_id`. Hit → return it.
2. Miss → check `ResponseCache`. Hit → return it.
3. Miss → generate once (same prompt as `generate_rules.py`), cache, return it.

Returns the single rule object from the schema above. Requires the same
`X-API-Key` header as the other endpoints.

## Client integration (sketch — not yet wired)

In `error_detail_sheet.dart`, add a "Learn more about this rule" action that
derives `rule_id` from the error (slug of the error topic/category), calls
`ApiService.fetchRule(...)`, shows the lesson + exercises, and caches the
fetched rule locally (SharedPreferences) so it works offline after first view.
