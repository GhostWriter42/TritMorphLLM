#!/usr/bin/env python3
"""Run real lm-evaluation-harness benchmarks with TritMorph wrapper."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, cast

from lm_eval import simple_evaluate

sys.path.insert(0, ".")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Import registers the custom model via decorator.
from eval.lm_eval_wrapper import TritMorphHarnessLM  # noqa: F401

RESULTS_DIR = ROOT / "results"
RESULTS_PATH = RESULTS_DIR / "standard_benchmarks.md"
TASKS = ["hellaswag", "arc_challenge", "winogrande", "piqa", "boolq", "humaneval"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run standard benchmark subset via lm-eval")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-type", type=str, default="tritmorph", choices=["tritmorph", "vanilla_bpe"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=float, default=500)
    parser.add_argument("--max-gen-toks", type=int, default=16)
    return parser.parse_args()


def pick_metric(task_name: str, metrics: dict[str, Any]) -> tuple[str, float]:
    preference = [
        "acc_norm,none",
        "acc,none",
        "exact_match,strict-match",
        "pass@1,create_test",
        "pass@1,none",
        "f1,none",
    ]
    for key in preference:
        if key in metrics and isinstance(metrics[key], (int, float)):
            metric = key.split(",", 1)[0]
            return metric, float(metrics[key])
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and not key.endswith("_stderr,none"):
            metric = key.split(",", 1)[0]
            return metric, float(value)
    raise RuntimeError(f"No numeric metric found for task: {task_name}")


def build_markdown(
    checkpoint: Path,
    model_type: str,
    limit: float,
    max_gen_toks: int,
    rows: list[tuple[str, str, float]],
) -> str:
    lines = [
        "# Standard Benchmark Results",
        "",
        f"Model type: `{model_type}`",
        f"Checkpoint: `{checkpoint}`",
        f"Task limit: `{int(limit) if float(limit).is_integer() else limit}` examples per task.",
        f"Generation cap for generative tasks: `{max_gen_toks}` tokens.",
        "",
        "| Task | Metric | Score |",
        "| --- | --- | ---: |",
    ]
    for task, metric, value in rows:
        lines.append(f"| {task} | {metric} | {value:.4f} |")
    return "\n".join(lines) + "\n"


def extract_rows(eval_results: dict[str, Any]) -> list[tuple[str, str, float]]:
    result_map = cast(dict[str, dict[str, Any]], eval_results.get("results") or {})
    rows: list[tuple[str, str, float]] = []
    for task in TASKS:
        if task not in result_map:
            continue
        metric, score = pick_metric(task, result_map[task])
        rows.append((task, metric, score))
    return rows


def main() -> None:
    args = parse_args()
    os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")
    model_args = {
        "checkpoint": str(args.checkpoint),
        "model_type": args.model_type,
        "device": args.device,
        "batch_size": args.batch_size,
        "max_gen_toks": args.max_gen_toks,
    }
    eval_results = cast(
        dict[str, Any],
        simple_evaluate(
        model="tritmorph",
        model_args=model_args,
        tasks=list(TASKS),
        limit=args.limit,
        log_samples=False,
        confirm_run_unsafe_code=True,
        gen_kwargs={"max_gen_toks": args.max_gen_toks},
        ),
    )

    rows = extract_rows(eval_results)

    markdown = build_markdown(args.checkpoint, args.model_type, args.limit, args.max_gen_toks, rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(markdown, encoding="utf-8")
    (RESULTS_DIR / "standard_benchmarks.json").write_text(json.dumps(eval_results, indent=2, default=str), encoding="utf-8")
    print(markdown)
    print("Paper-ready results generated from the real lm_eval wrapper.")


if __name__ == "__main__":
    main()
