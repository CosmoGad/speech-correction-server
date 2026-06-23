# Dynamic rule book — design spec (resolver + lazy generation)

Status: **spec only, not yet implemented.** Builds on the static rule book
(see `README.md`). Goal: the rule book "grows with use" — a learner taps
"Learn more about this rule" on a correction and gets the right lesson, even if
it wasn't pre-generated — **without ever creating duplicate rules**.

## Core principle (the anti-duplication guarantee)

> A `rule_id` is **never minted from the free text of an error**. It is only ever
> **selected from a fixed, curated taxonomy** per learning language.

That single rule is what makes duplicates impossible by design. Everything below
is built around it. The taxonomy = the existing `topics_<learning>.json` registry,
treated as a controlled vocabulary (MECE-ish: each key is a distinct concept).
The taxonomy grows only through a **deliberate, reviewed** batch process — never
automatically from individual errors.

## Pieces

| Piece | Where | Role |
|-------|-------|------|
| Taxonomy / registry | `rules/topics_<learning>.json` | the only source of `rule_id`s (per learning lang, interface-independent) |
| Static content | `rules/rules_<learning>_<interface>.json` | pre-generated lessons |
| On-demand content | `ResponseCache` (SQLite) | lessons generated lazily on first request |
| Error→rule map cache | `ResponseCache` | `hash(error signature) → rule_id` so we resolve each distinct error once |

## Flow A — resolve an error to a rule (`GET/POST /resolve-rule`)

Triggered when the client taps "Learn more about this rule" on a correction.

Input: `learning`, `interface`, and the error (`type`, `original`, `corrected`,
`explanation`).

```
1. sig = sha256(learning + normalized(original) + normalized(corrected) + type)
2. if cache has sig → return {rule_id, matched:true}            # 0 model calls
3. load taxonomy = topics_<learning>.json  (list of {rule_id, title})
4. ask the model:
     "Here is an error a learner made: <error>.
      Here is the list of rule topics: <titles>.
      Reply with ONLY the single best-matching rule_id, or 'none'."
5. validate the reply IS an existing rule_id (else treat as 'none')
6. cache sig → rule_id (or 'none')
7. return {rule_id | null, matched}
```

Key safeguards:
- Step 5 (**reply must be an existing id**) is critical — the model can only
  *pick*, never *invent*. If it returns garbage or 'none', no rule is created.
- Resolving is cached per distinct error, so repeated/identical errors cost 0.
- `none` is cached too (negative cache) → unmatched errors don't re-call.

Alternative (rejected for now): tag each error with a `rule_id` inside the main
analysis prompt. Avoids the extra call but adds tokens to **every** analysis
(even when the user never opens a rule). The lazy resolver only spends on tap →
cheaper for our usage. Revisit if "Learn more" tap-rate gets high.

## Flow B — fetch/generate the lesson (`GET /rule`, extended)

`GET /rule?learning=&interface=&rule_id=` (rule_id always comes from Flow A or
the static list, i.e. always a valid taxonomy key):

```
1. static file rules_<learning>_<interface>.json has rule_id?  → return it
2. ResponseCache has (learning, interface, rule_id)?           → return it
3. generate lesson for that taxonomy key (same prompt as generate_rules.py),
   store in ResponseCache, return it
```

Because `rule_id` is always a taxonomy key, step 3 can never create a duplicate —
it only fills in missing content for an already-known concept. "Grows with use"
= content is generated lazily on first request, not that the taxonomy grows.

## Flow C — grow the taxonomy (controlled, offline, reviewed)

The taxonomy must occasionally gain genuinely new concepts — but deliberately:

```
1. collect signatures the resolver returned 'none' for (log/table)
2. periodically: feed those unmatched errors + existing taxonomy titles to the
   model → "propose new distinct rule topics not already covered"
3. dedup proposals against the registry (existing add_topics logic)
4. HUMAN REVIEW the proposed additions
5. add approved topics to topics_<learning>.json (+ optionally batch-generate
   their content)
```

This is the only path that adds `rule_id`s, and it has a human gate. Auto-adding
from errors is explicitly **out of scope** (that's what causes fragmentation).

## Duplication: is it solved?

- **Within static batch:** yes — registry + `rule_id` + normalized-title dedup +
  idempotent generation (already implemented).
- **Dynamic:** yes, *by construction* — ids only ever come from the fixed
  taxonomy (Flow A step 5, Flow B always-valid-key). No free-text id minting.
- **Semantic near-dups** (e.g. "Articles" vs "Definite vs zero article"): handled
  at taxonomy-creation time — the resolver maps errors onto whichever existing
  key is closest, and new keys pass through human review (Flow C step 4). The
  model seeing the whole taxonomy is what catches "this is basically rule X".

## Caching summary

| Cache | Key | Value | TTL |
|-------|-----|-------|-----|
| error→rule | sha256(learning,original,corrected,type) | rule_id or 'none' | 30d (reuse ResponseCache) |
| lesson | (learning, interface, rule_id) | full rule JSON | 30d |

Both ephemeral on fly (refill after deploy) — acceptable, same as the existing
response cache.

## Client changes

- "Learn more about this rule" button (already exists in `error_detail_sheet.dart`)
  → call `/resolve-rule` with the current error → if `rule_id`, open
  `RuleDetailScreen(rule_id)`; if `null`, fall back to the rule **list** (current
  behaviour) or a gentle "no specific rule yet" note.
- `RulesService`: add `resolveRule(...)`; `fetchRule` already handles a missing
  rule via the server's lazy generation (Flow B) transparently.

## Token / cost

- Analysis prompt: **unchanged** (no per-analysis cost added).
- Per distinct unmatched error: 1 small resolve call (cached forever after).
- Per never-before-seen rule+interface: 1 generation call (cached).
- Everything else: 0 model calls (served from file/cache).

## Open decisions

1. Resolver model: keep DeepSeek (free promo now); after 28 Jun pick a cheap model.
2. Negative-cache TTL for 'none' (so newly added taxonomy keys eventually re-match)
   — maybe shorter than 30d, or invalidate on taxonomy growth.
3. Whether to expose Flow C's "unmatched" collection as a tiny admin dump or just
   log lines to grep.
