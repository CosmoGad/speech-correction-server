# Text-analysis V2 evals

`scenarios.json` is the fixed behavior baseline for prompt changes. Validate
recorded responses offline:

```powershell
python evals/text_analysis/run_eval.py --input recorded-v2.json --output eval-report.json --require-language-judgement
```

Run the complete live suite only from a controlled environment with an existing
provider key. It writes a reviewable report with tokens and latency, but never
uses the learner cache:

```powershell
python evals/text_analysis/run_eval.py --live --output evals/text_analysis/results/v2-$(Get-Date -Format yyyyMMdd).json --require-language-judgement
```

The runner checks the strict V2 schema, expected outcome, category allow-list,
correction invariants, complete scenario coverage, language judgement, latency,
token usage and a cost estimate. Its report includes model responses only for
the committed synthetic corpus; it never writes learner input or uses the
learner cache. Live results are operational artifacts and are ignored by Git.
