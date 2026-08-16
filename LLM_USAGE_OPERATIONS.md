# LLM usage telemetry

`llm_usage.db` contains only daily aggregates grouped by product feature and
model. It never stores a user ID, IP address, cache key, prompt, response, or
learner text.

The `llm_usage_daily` table records model calls, cache hits, prompt/output
tokens, cumulative latency, and an optional cost estimate in microdollars.
Set the following Fly secrets only after confirming the provider's current
prices:

```powershell
flyctl secrets set LLM_INPUT_COST_PER_MTOK=<current-input-price> LLM_OUTPUT_COST_PER_MTOK=<current-output-price> --app speech-correction
```

Leaving both variables at their default of `0` keeps token and latency metrics
while deliberately reporting no invented price. The file is a local operational
artifact and is excluded from Git.
