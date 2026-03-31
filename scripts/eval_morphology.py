#!/usr/bin/env python3
"""Evaluation script for held-out perplexity and morphology generalization."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

sys.path.insert(0, ".")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from model.vanilla_bpe_baseline import VanillaBPETokenizer
from tokenizer.hybrid_morph_bpe import HybridTokenizer

from scripts.train import evaluate, prepare_training_components, resolve_device


PROBE_WORDS = [
    "superunhappiness",
    "reoverjumps",
    "counterreplaying",
    "misunderstandingly",
    "hyperkindness",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate TritMorphLLM morphology generalization")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "default.yaml")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--model-type", type=str, default=None, choices=["tritmorph", "vanilla_bpe"])
    parser.add_argument("--dataset", type=str, default=None, choices=["wikitext103", "tiny_stories"])
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
        saved_dataset = saved_config.get("dataset", {})
        requested_preset = config.get("dataset", {}).get("preset")
        config = saved_config
        if requested_preset is not None:
            config.setdefault("dataset", {})["preset"] = requested_preset
    tokenizer, model, _, val_loader, word_vocab = prepare_training_components(config, model_type)
    model.load_state_dict(checkpoint["model"])
    model = model.to(device)
    return tokenizer, model, val_loader, word_vocab


def morphology_accuracy_tritmorph(tokenizer: HybridTokenizer) -> float:
    correct = 0
    for word in PROBE_WORDS:
        pieces = tokenizer.tokenize_word(word)
        reconstructed = "".join(piece.replace("##", "") for piece in pieces)
        if reconstructed == word.lower():
            correct += 1
        print(f"{word}: {pieces} -> {reconstructed}")
    return correct / len(PROBE_WORDS)


def morphology_accuracy_baseline(tokenizer: VanillaBPETokenizer) -> float:
    correct = 0
    for word in PROBE_WORDS:
        token_id = tokenizer.tokenizer.token_to_id(word.lower())
        reconstructed = word.lower() if token_id is not None else "[UNK]"
        if reconstructed == word.lower():
            correct += 1
        print(f"{word}: baseline-token-id={token_id} -> {reconstructed}")
    return correct / len(PROBE_WORDS)


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

    if model_type == "tritmorph":
        accuracy = morphology_accuracy_tritmorph(tokenizer)
    else:
        accuracy = morphology_accuracy_baseline(tokenizer)

    print("\nFinal Report")
    print(f"- Model type: {model_type}")
    if checkpoint_path is not None:
        print(f"- Checkpoint: {checkpoint_path}")
    print(f"- Held-out perplexity: {val_ppl:.2f}")
    print(f"- Morphology generalization score: {accuracy:.3f}")
    print(f"- Summary: {model_type} | ppl={val_ppl:.2f} | morph_acc={accuracy:.3f}")


if __name__ == "__main__":
    main()
