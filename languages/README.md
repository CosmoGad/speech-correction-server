# Languages catalog v2

## Goal

Catalog v2 replaces independent language registries and copied prompts with one
declarative source of truth. Adding a language must be a data and localization
change, not conditional Python or Flutter code.

`catalog.v2.json` will keep a canonical code, native and prompt names,
`detection_code`, text direction, roles (learning/interface), context templates
and interface labels for every language/locale. Level descriptions use those
canonical codes, never display names. A single versioned prompt template uses
catalog data; it does not have per-language policy copies.

## Safe migration

1. Mechanically create `catalog.v2.json`, including explicit `detection_code`
   (for example, `es_MX` -> `es`) and legacy aliases for released clients.
2. Validate it at server startup and expose only compatibility *views* from it;
   no endpoint reads an independent language list.
3. Keep `language_catalog_validator.py` in CI while legacy prompt files are
   still migration inputs; parity tests protect the initial conversion.
4. Compare the V1 and V2 analysis paths through the committed eval set. Remove
   legacy prompt/configuration inputs only after the observation period.
