#!/usr/bin/env python3
"""Automated experiment runner for TritMorphLLM Phase 4."""

from __future__ import annotations

import argparse
import glob
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

sys.path.insert(0, ".")

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
RESULTS_PATH = RESULTS_DIR / "experiment_results.md"


@dataclass(slots=True)
class ExperimentResult:
    model_name: str
    model_type: str
    checkpoint_path: Path
    perplexity: float
    morph_acc: float
    training_time_sec: float
    gpu_memory_mb: float
    train_log_path: Path
    eval_log_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TritMorphLLM vs Vanilla BPE experiment")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "default.yaml")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--dataset", type=str, default=None, choices=["wikitext103", "tiny_stories"])
    parser.add_argument("--resume-tritmorph-from", type=Path, default=None)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def get_gpu_memory_mb() -> float:
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip() or "0"
    try:
        command = [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
            "-i",
            cuda_visible,
        ]
        output = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True)
        return float(output.stdout.strip().splitlines()[0])
    except Exception:
        if torch is not None and torch.cuda.is_available():
            try:
                return float(torch.cuda.max_memory_allocated() / (1024 * 1024))
            except Exception:
                return 0.0
        return 0.0


def run_command(command: list[str], log_path: Path) -> tuple[float, str]:
    start = time.perf_counter()
    process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True)
    elapsed = time.perf_counter() - start
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(process.stdout + ("\n" + process.stderr if process.stderr else ""), encoding="utf-8")
    return elapsed, process.stdout


def extract_metric(pattern: str, text: str) -> float:
    match = re.search(pattern, text)
    if match is None:
        raise RuntimeError(f"Failed to parse metric with pattern: {pattern}")
    return float(match.group(1))


def format_duration(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    return f"{minutes}m {secs}s"


def build_markdown(results: list[ExperimentResult], max_steps: int, dataset_name: str) -> str:
    lines = [
        "# Experiment Results",
        "",
        f"Generated for `max_steps={max_steps}`.",
        f"Dataset preset: `{dataset_name}`.",
        "",
        "| Model | Held-out Perplexity | Morph Acc | Training Time | GPU Memory (MB) | Checkpoint |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for result in results:
        lines.append(
            f"| {result.model_name} | {result.perplexity:.2f} | {result.morph_acc:.3f} | "
            f"{format_duration(result.training_time_sec)} | {result.gpu_memory_mb:.0f} | `{result.checkpoint_path}` |"
        )

    tritmorph = next(result for result in results if result.model_type == "tritmorph")
    vanilla = next(result for result in results if result.model_type == "vanilla_bpe")
    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"TritMorph vs Vanilla: Perplexity {tritmorph.perplexity:.2f}->{vanilla.perplexity:.2f} | "
            f"Morph Gen {tritmorph.morph_acc * 100:.1f}%->{vanilla.morph_acc * 100:.1f}%",
            "",
            "## Raw Logs",
            "",
            f"- TritMorph train log: `{tritmorph.train_log_path}`",
            f"- TritMorph eval log: `{tritmorph.eval_log_path}`",
            f"- Vanilla train log: `{vanilla.train_log_path}`",
            f"- Vanilla eval log: `{vanilla.eval_log_path}`",
            f"- Detailed morphology probe CSV: `{ROOT / 'results' / 'morphology_probe_detailed.csv'}`",
        ]
    )
    return "\n".join(lines) + "\n"


def infer_training_time_from_log(log_path: Path) -> float:
    if not log_path.exists():
        return 0.0
    content = log_path.read_text(encoding="utf-8")
    match = re.search(r"training:[^\n]+100%\|[^\n]*\[(\d+):(\d+)<", content)
    if match is not None:
        minutes = int(match.group(1))
        seconds = int(match.group(2))
        return float(minutes * 60 + seconds)
    return 0.0


def resolve_latest_checkpoint(output_dir: Path, model_type: str, max_steps: int, preferred_path: Path | None = None) -> Path:
    if preferred_path is not None and preferred_path.exists():
        return preferred_path
    expected = output_dir / f"{model_type}_step_{max_steps}.pt"
    if expected.exists():
        return expected
    candidates = sorted(glob.glob(str(output_dir / f"{model_type}_step_*.pt")))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found for {model_type} in {output_dir}")
    return Path(candidates[-1])


def run_single_experiment(
    config: Path,
    model_type: str,
    output_dir: Path,
    max_steps: int,
    use_ternary: bool,
    dataset_name: str | None,
    resume_from: Path | None = None,
) -> ExperimentResult:
    train_log = RESULTS_DIR / f"{model_type}_train.log"
    eval_log = RESULTS_DIR / f"{model_type}_eval.log"

    train_command = [
        sys.executable,
        "-m",
        "scripts.train",
        "--config",
        str(config),
        "--output-dir",
        str(output_dir),
        "--max-steps",
        str(max_steps),
        "--model-type",
        model_type,
    ]
    if dataset_name is not None:
        train_command.extend(["--dataset", dataset_name])
    if use_ternary:
        train_command.extend(["--use-ternary", "true"])
    if resume_from is not None:
        train_command.extend(["--resume-from", str(resume_from)])

    checkpoint_path = resolve_latest_checkpoint(
        output_dir,
        model_type,
        max_steps,
        preferred_path=resume_from,
    ) if (resume_from is not None or output_dir.exists()) else output_dir / f"{model_type}_step_{max_steps}.pt"
    if resume_from is not None:
        training_time = 0.0
    elif checkpoint_path.exists():
        training_time = infer_training_time_from_log(train_log)
    else:
        training_time, _ = run_command(train_command, train_log)
        checkpoint_path = resolve_latest_checkpoint(output_dir, model_type, max_steps)
    gpu_memory_mb = get_gpu_memory_mb()

    eval_command = [
        sys.executable,
        "-m",
        "scripts.eval_morphology",
        "--config",
        str(config),
        "--checkpoint",
        str(checkpoint_path),
        "--model-type",
        model_type,
        "--step",
        str(max_steps),
    ]
    if dataset_name is not None:
        eval_command.extend(["--dataset", dataset_name])
    _, eval_stdout = run_command(eval_command, eval_log)

    perplexity = extract_metric(r"Held-out perplexity: ([0-9.]+)", eval_stdout)
    morph_acc = extract_metric(r"Morphology generalization score: ([0-9.]+)", eval_stdout)

    return ExperimentResult(
        model_name="TritMorph (ternary)" if model_type == "tritmorph" else "Vanilla BPE",
        model_type=model_type,
        checkpoint_path=checkpoint_path,
        perplexity=perplexity,
        morph_acc=morph_acc,
        training_time_sec=training_time,
        gpu_memory_mb=gpu_memory_mb,
        train_log_path=train_log,
        eval_log_path=eval_log,
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    max_steps = args.max_steps or int(config["training"]["max_steps"])
    dataset_name = args.dataset or config["dataset"].get("preset", "wikitext103")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    tritmorph_result = run_single_experiment(
        config=args.config,
        model_type="tritmorph",
        output_dir=ROOT / "checkpoints" / "tritmorph_ternary",
        max_steps=max_steps,
        use_ternary=True,
        dataset_name=dataset_name,
        resume_from=args.resume_tritmorph_from,
    )
    vanilla_result = run_single_experiment(
        config=args.config,
        model_type="vanilla_bpe",
        output_dir=ROOT / "checkpoints" / "vanilla_bpe",
        max_steps=max_steps,
        use_ternary=False,
        dataset_name=dataset_name,
    )

    markdown = build_markdown([tritmorph_result, vanilla_result], max_steps=max_steps, dataset_name=dataset_name)
    RESULTS_PATH.write_text(markdown, encoding="utf-8")

    print(
        f"TritMorph vs Vanilla: Perplexity {tritmorph_result.perplexity:.2f}->{vanilla_result.perplexity:.2f} | "
        f"Morph Gen {tritmorph_result.morph_acc * 100:.1f}%->{vanilla_result.morph_acc * 100:.1f}%"
    )
    print(f"GPU memory | TritMorph: {tritmorph_result.gpu_memory_mb:.0f} MB | Vanilla: {vanilla_result.gpu_memory_mb:.0f} MB")
    print(f"Saved markdown report to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
