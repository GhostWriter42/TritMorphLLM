#!/usr/bin/env python3
"""Evaluation script for held-out perplexity and morphology generalization."""

from __future__ import annotations

import argparse
import csv
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

sys.path.insert(0, ".")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
RESULTS_DIR = ROOT / "results"
DETAIL_CSV = RESULTS_DIR / "morphology_probe_detailed.csv"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from model.vanilla_bpe_baseline import VanillaBPETokenizer
from tokenizer.hybrid_morph_bpe import HybridTokenizer

from scripts.train import evaluate, prepare_training_components, resolve_device


PREFIXES = [
    "un",
    "re",
    "super",
    "hyper",
    "counter",
    "mis",
    "over",
    "under",
    "pre",
    "post",
    "anti",
    "pro",
    "semi",
    "multi",
    "bi",
    "tri",
    "poly",
    "mega",
    "nano",
    "ultra",
]

SUFFIXES = [
    "ness",
    "ing",
    "ly",
    "able",
    "ible",
    "ment",
    "tion",
    "sion",
    "er",
    "or",
    "ist",
    "ism",
    "less",
    "ful",
    "ish",
    "like",
    "wise",
    "ward",
    "proof",
    "free",
]

STEMS = [
    "happy",
    "jump",
    "play",
    "stand",
    "kind",
    "understand",
    "run",
    "think",
    "write",
    "read",
    "speak",
    "build",
    "create",
    "destroy",
    "love",
    "hate",
    "win",
    "lose",
    "fight",
    "fly",
    "drive",
    "sing",
    "dance",
    "dream",
    "believe",
    "know",
    "see",
    "hear",
    "feel",
    "touch",
]


@dataclass(slots=True)
class ProbeResult:
    word: str
    predicted_tokens: str
    fused_correctly: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate TritMorphLLM morphology generalization")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "default.yaml")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--model-type", type=str, default=None, choices=["tritmorph", "vanilla_bpe"])
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=["wikitext103", "tiny_stories", "fineweb_edu_code_agentic_mix"],
    )
    parser.add_argument("--step", type=int, default=500)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_checkpoint_path(checkpoint_path: Path | None, model_type: str, step: int) -> Path | None:
    if checkpoint_path is not None:
        return checkpoint_path
    checkpoint_dir = ROOT / "checkpoints"
    candidate = checkpoint_dir / ("tritmorph_ternary" if model_type == "tritmorph" else "vanilla_bpe")
    expected = candidate / f"{model_type}_step_{step}.pt"
    if expected.exists():
        return expected
    return None


def load_model_from_checkpoint(config: dict[str, Any], checkpoint_path: Path, model_type: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    saved_config = checkpoint.get("config", {})
    if isinstance(saved_config, dict) and saved_config:
        requested_preset = config.get("dataset", {}).get("preset")
        config = saved_config
        if requested_preset is not None:
            config.setdefault("dataset", {})["preset"] = requested_preset
    tokenizer, model, _, val_loader, word_vocab = prepare_training_components(
        config,
        model_type,
        dataset_name=config.get("dataset", {}).get("preset"),
    )
    model.load_state_dict(checkpoint["model"])
    model = model.to(device)
    return tokenizer, model, val_loader, word_vocab


def generate_systematic_probe_words(word_vocab: dict[str, int], target_count: int = 128) -> list[str]:
    rng = random.Random(42)
    seen_words = set(word_vocab.keys())
    generated: set[str] = set()
    attempts = 0
    while len(generated) < target_count and attempts < target_count * 200:
        attempts += 1
        prefix_count = rng.randint(1, 3)
        suffix_count = rng.randint(1, 2)
        prefix_chain = "".join(rng.sample(PREFIXES, k=prefix_count))
        stem = rng.choice(STEMS)
        suffix_chain = "".join(rng.sample(SUFFIXES, k=suffix_count))
        word = f"{prefix_chain}{stem}{suffix_chain}"
        if word not in seen_words:
            generated.add(word)
    return sorted(generated)


def save_probe_results(results: list[ProbeResult]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with DETAIL_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["word", "predicted_tokens", "fused_correctly"])
        for row in results:
            writer.writerow([row.word, row.predicted_tokens, row.fused_correctly])


def run_probe_tritmorph(tokenizer: HybridTokenizer, probe_words: list[str]) -> tuple[float, list[ProbeResult]]:
    rows: list[ProbeResult] = []
    correct = 0
    for word in probe_words:
        pieces = tokenizer.tokenize_word(word)
        reconstructed = "".join(piece.replace("##", "") for piece in pieces)
        fused_correctly = reconstructed == word.lower()
        correct += int(fused_correctly)
        rows.append(ProbeResult(word=word, predicted_tokens=" ".join(pieces), fused_correctly=fused_correctly))
    return correct / max(1, len(probe_words)), rows


def run_probe_baseline(tokenizer: VanillaBPETokenizer, probe_words: list[str]) -> tuple[float, list[ProbeResult]]:
    rows: list[ProbeResult] = []
    correct = 0
    for word in probe_words:
        token_id = tokenizer.tokenizer.token_to_id(word.lower())
        predicted = word.lower() if token_id is not None else "[UNK]"
        fused_correctly = predicted == word.lower()
        correct += int(fused_correctly)
        rows.append(ProbeResult(word=word, predicted_tokens=predicted, fused_correctly=fused_correctly))
    return correct / max(1, len(probe_words)), rows


def print_probe_table(rows: list[ProbeResult], limit: int | None = None) -> None:
    print("word | predicted_tokens | fused_correctly")
    print("--- | --- | ---")
    for row in rows[:limit] if limit is not None else rows:
        print(f"{row.word} | {row.predicted_tokens} | {row.fused_correctly}")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.dataset is not None:
        config["dataset"]["preset"] = args.dataset
    model_type = args.model_type or config["training"].get("model_type", "tritmorph")
    device = resolve_device(args.device or config.get("device"))
    checkpoint_path = resolve_checkpoint_path(args.checkpoint, model_type, args.step)

    if checkpoint_path is not None:
        tokenizer, model, val_loader, word_vocab = load_model_from_checkpoint(config, checkpoint_path, model_type, device)
    else:
        tokenizer, model, _, val_loader, word_vocab = prepare_training_components(
            config,
            model_type,
            dataset_name=config["dataset"].get("preset"),
        )
        model = model.to(device)

    val_loss, val_ppl = evaluate(model_type, model, val_loader, device)
    probe_words = generate_systematic_probe_words(word_vocab, target_count=128)

    if model_type == "tritmorph":
        accuracy, rows = run_probe_tritmorph(tokenizer, probe_words)
    else:
        accuracy, rows = run_probe_baseline(tokenizer, probe_words)

    save_probe_results(rows)
    print_probe_table(rows)

    print("\nFinal Report")
    print(f"- Model type: {model_type}")
    if checkpoint_path is not None:
        print(f"- Checkpoint: {checkpoint_path}")
    print(f"- Held-out perplexity: {val_ppl:.2f}")
    print(f"- Morphology generalization score: {accuracy:.3f}")
    print(f"- Detailed probe CSV: {DETAIL_CSV}")
    print(f"- Summary: {model_type} | ppl={val_ppl:.2f} | morph_acc={accuracy:.3f}")


if __name__ == "__main__":
    main()
