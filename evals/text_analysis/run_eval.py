"""Validate recorded V2 model responses against the committed eval set.

This runner never calls a model.  A provider adapter records responses in the
documented input format, and CI can then apply deterministic contract and
scenario checks before a prompt change is released.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
SERVER_ROOT = ROOT.parents[1]
if str(SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVER_ROOT))

from openai import AsyncOpenAI, AuthenticationError

from prompt_contract_v2 import AnalysisOutput


SCENARIOS_PATH = ROOT / "scenarios.json"
DEFAULT_RESULTS_DIR = ROOT / "results"


@dataclass(frozen=True)
class EvalFailure:
    scenario_id: str
    reason: str


def evaluate_records(
    scenarios: list[dict[str, Any]],
    records: list[dict[str, Any]],
    *,
    require_language_judgement: bool = False,
) -> list[EvalFailure]:
    """Return all deterministic contract failures for a complete eval run."""
    expected_by_id = {scenario["id"]: scenario for scenario in scenarios}
    records_by_id: dict[str, dict[str, Any]] = {}
    failures: list[EvalFailure] = []
    for record in records:
        scenario_id = record.get("scenario_id") if isinstance(record, dict) else None
        if not isinstance(scenario_id, str):
            failures.append(EvalFailure("<unknown>", "record must contain string scenario_id"))
            continue
        if scenario_id in records_by_id:
            failures.append(EvalFailure(scenario_id, "duplicate recorded response"))
            continue
        records_by_id[scenario_id] = record
    for scenario_id in sorted(expected_by_id):
        scenario = expected_by_id[scenario_id]
        record = records_by_id.pop(scenario_id, None)
        if record is None:
            failures.append(EvalFailure(scenario_id, "missing recorded response"))
            continue
        failures.extend(_evaluate_one(scenario, record, require_language_judgement))
    for scenario_id in sorted(records_by_id):
        failures.append(EvalFailure(scenario_id, "response has no committed scenario"))
    return failures


def _evaluate_one(
    scenario: dict[str, Any], record: dict[str, Any], require_language_judgement: bool,
) -> list[EvalFailure]:
    scenario_id = scenario["id"]
    failures: list[EvalFailure] = []
    if require_language_judgement and record.get("language_judgement") is not True:
        failures.append(EvalFailure(scenario_id, "missing or failed explanation-language judgement"))
    try:
        output = AnalysisOutput.model_validate(record.get("response"))
    except Exception as error:
        return failures + [EvalFailure(scenario_id, f"invalid V2 response: {error}")]

    request = scenario["request"]
    expected = scenario["expected"]
    text = request["text"]
    if expected["outcome"] == "clean":
        if output.errors:
            failures.append(EvalFailure(scenario_id, "clean scenario contains errors"))
        if output.corrected_text != text:
            failures.append(EvalFailure(scenario_id, "clean scenario changed submitted text"))
        return failures

    if len(output.errors) < expected["min_error_count"]:
        failures.append(EvalFailure(scenario_id, "fewer errors than expected"))
    allowed_categories = set(expected["allowed_error_types"])
    for error in output.errors:
        if error.category.value not in allowed_categories:
            failures.append(EvalFailure(scenario_id, f"unexpected category {error.category.value}"))
        if error.original not in text:
            failures.append(EvalFailure(scenario_id, "error original is not an input substring"))
        if error.corrected not in output.corrected_text:
            failures.append(EvalFailure(scenario_id, "error corrected is not a corrected-text substring"))
    return failures


def _read_list(path: Path, label: str) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"{label} must be a JSON array of objects")
    return value


async def collect_live_records(scenarios: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Run the committed scenarios through V2 and retain no user-originated data.

    The scenarios are intentionally synthetic product fixtures. Results are
    written only when the caller explicitly supplies an output path; the runner
    never touches the server's learner cache.
    """
    # Import lazily so offline validation remains light-weight and so .env is
    # loaded only for an explicit paid live run.
    import speech_correction_server as server

    key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY or OPENAI_API_KEY is required for --live")
    client = AsyncOpenAI(
        api_key=key,
        base_url="https://api.deepseek.com/v1",
        timeout=30.0,
    )
    records: list[dict[str, Any]] = []
    try:
        for scenario in scenarios:
            request = server.CorrectionRequest(**scenario["request"])
            system_prompt, user_payload = server.build_v2_prompt(request)
            started = time.perf_counter()
            try:
                response = await client.chat.completions.create(
                    model=server.DEEPSEEK_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_payload},
                    ],
                    temperature=0.3,
                    max_tokens=server.DEEPSEEK_MAX_TOKENS,
                    response_format={"type": "json_object"},
                    extra_body={"thinking": server.DEEPSEEK_THINKING},
                )
                raw = response.choices[0].message.content or "{}"
                parsed = server._extract_json_object(raw)
                output = AnalysisOutput.model_validate(parsed)
                expected_code = server.INTERFACE_LANGUAGES[request.interface_language]["language_code"]
                texts = [output.summary, *(error.explanation for error in output.errors)]
                language_judgement = all(
                    not server._is_wrong_language(text, expected_code) for text in texts if text
                )
                usage = getattr(response, "usage", None)
                records.append({
                    "scenario_id": scenario["id"],
                    "response": parsed,
                    "language_judgement": language_judgement,
                    "metrics": {
                        "model": server.DEEPSEEK_MODEL,
                        "latency_ms": round((time.perf_counter() - started) * 1000),
                        "prompt_tokens": getattr(usage, "prompt_tokens", None),
                        "completion_tokens": getattr(usage, "completion_tokens", None),
                    },
                })
            except AuthenticationError as error:
                raise RuntimeError(
                    "Provider authentication failed; update the controlled eval credential"
                ) from error
            except Exception as error:
                records.append({
                    "scenario_id": scenario["id"],
                    "response": {},
                    "language_judgement": False,
                    "metrics": {"latency_ms": round((time.perf_counter() - started) * 1000)},
                    "runner_error": type(error).__name__,
                })
    finally:
        await client.close()
    return records


def _metrics_summary(records: list[dict[str, Any]]) -> dict[str, int | float]:
    metrics = [record.get("metrics", {}) for record in records]
    latencies = [value for value in (metric.get("latency_ms") for metric in metrics) if isinstance(value, int)]
    prompt_tokens = [value for value in (metric.get("prompt_tokens") for metric in metrics) if isinstance(value, int)]
    completion_tokens = [value for value in (metric.get("completion_tokens") for metric in metrics) if isinstance(value, int)]
    input_price = float(os.getenv("LLM_INPUT_COST_PER_MTOK", "0"))
    output_price = float(os.getenv("LLM_OUTPUT_COST_PER_MTOK", "0"))
    if input_price < 0 or output_price < 0:
        raise ValueError("LLM token prices must be non-negative")
    estimated_cost_usd = (
        sum(prompt_tokens) * input_price + sum(completion_tokens) * output_price
    ) / 1_000_000
    return {
        "scenario_count": len(records),
        "successful_model_responses": sum("runner_error" not in record for record in records),
        "total_prompt_tokens": sum(prompt_tokens),
        "total_completion_tokens": sum(completion_tokens),
        "average_latency_ms": round(sum(latencies) / len(latencies)) if latencies else 0,
        "p95_latency_ms": sorted(latencies)[max(0, round(len(latencies) * 0.95) - 1)] if latencies else 0,
        "estimated_cost_usd": round(estimated_cost_usd, 8),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate recorded text-analysis V2 eval responses.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", type=Path, help="Recorded responses JSON.")
    input_group.add_argument("--live", action="store_true", help="Call the configured provider for every scenario.")
    parser.add_argument("--output", type=Path, help="Required for --live; optional offline JSON report path.")
    parser.add_argument("--require-language-judgement", action="store_true")
    args = parser.parse_args()
    scenarios = _read_list(SCENARIOS_PATH, "scenarios")
    if args.live and args.output is None:
        parser.error("--live requires --output so results are reviewable")
    records = (
        asyncio.run(collect_live_records(scenarios))
        if args.live
        else _read_list(args.input, "records")
    )
    failures = evaluate_records(
        scenarios,
        records,
        require_language_judgement=args.require_language_judgement,
    )
    report = {
        "passed": not failures,
        "failure_count": len(failures),
        "failures": [failure.__dict__ for failure in failures],
        "metrics": _metrics_summary(records),
        # The corpus is committed synthetic test data, never learner input.  Keeping
        # responses alongside the verdict makes a failed model run reproducible.
        "records": records,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
