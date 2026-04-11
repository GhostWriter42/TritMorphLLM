#!/usr/bin/env python3
"""Real decoding speed test for TritMorph and Vanilla models."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import Any

import torch
import yaml

sys.path.insert(0, ".")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from model.tritmorph_model import TritMorphConfig, TritMorphModel
from model.vanilla_bpe_baseline import VanillaBPEBaseline, VanillaBPEConfig
from tokenizer.hybrid_morph_bpe import HybridTokenizer

RESULTS_DIR = ROOT / "results"
RESULTS_PATH = RESULTS_DIR / "speed_test.md"


@dataclass(slots=True)
class SpeedRow:
    mode: str
    device: str
    ternary: str
    tokens_per_second: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real speed evaluation")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-type", type=str, default="tritmorph", choices=["tritmorph", "vanilla_bpe"])
    parser.add_argument("--input-tokens", type=int, default=512)
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_model_from_checkpoint(checkpoint_path: Path, model_type: str, device: torch.device, force_ternary: bool):
    payload = torch.load(checkpoint_path, map_location=device)
    config = payload["config"]
    model_cfg = dict(config["model"])
    model_cfg["use_ternary"] = force_ternary

    output_dir = checkpoint_path.parent
    tokenizer_path = output_dir / f"{model_type}_tokenizer.json"
    if not tokenizer_path.exists():
        tokenizer_path = output_dir / "tritmorph_tokenizer.json"
    tokenizer = HybridTokenizer.from_file(tokenizer_path)

    if model_type == "tritmorph":
        model = TritMorphModel(TritMorphConfig(**model_cfg)).to(device)
    else:
        word_vocab_path = output_dir / f"{model_type}_word_vocab.json"
        if not word_vocab_path.exists():
            word_vocab_path = output_dir / "tritmorph_word_vocab.json"
        import json

        word_vocab = json.loads(word_vocab_path.read_text(encoding="utf-8"))
        baseline_cfg = VanillaBPEConfig(
            vocab_size=model_cfg["vocab_size"],
            word_vocab_size=len(word_vocab),
            max_position_embeddings=model_cfg["max_position_embeddings"],
            d_model=model_cfg["d_model"],
            n_heads=model_cfg["n_heads"],
            n_layers=model_cfg["n_layers"],
            mlp_ratio=model_cfg["mlp_ratio"],
            dropout=model_cfg["dropout"],
            pad_token_id=model_cfg["pad_token_id"],
            use_ternary=force_ternary,
        )
        model = VanillaBPEBaseline(baseline_cfg).to(device)
    model.load_state_dict(payload["model"], strict=False)
    model.eval()
    return model, tokenizer


def next_logits_tritmorph(model: TritMorphModel, tokenizer: HybridTokenizer, text: str, device: torch.device) -> torch.Tensor:
    words = [w for w in text.split(" ") if w]
    max_words = model.config.max_position_embeddings
    if len(words) > max_words:
        text = " ".join(words[-max_words:])
    encoded = tokenizer.encode(text, add_special_tokens=False)
    if not encoded.input_ids:
        encoded = tokenizer.encode("[UNK]", add_special_tokens=False)
    if len(encoded.word_spans) > model.config.max_position_embeddings:
        encoded.word_spans = encoded.word_spans[-model.config.max_position_embeddings :]
        start = encoded.word_spans[0].start
        encoded.input_ids = encoded.input_ids[start:]
        from tokenizer.hybrid_morph_bpe import WordSpan

        encoded.word_spans = [
            WordSpan(word=span.word, start=span.start - start, end=span.end - start) for span in encoded.word_spans
        ]
    spans = torch.full((1, max(1, len(encoded.word_spans)), 2), -1, dtype=torch.long, device=device)
    for idx, span in enumerate(encoded.word_spans):
        spans[0, idx, 0] = span.start
        spans[0, idx, 1] = span.end
    input_ids = torch.tensor([encoded.input_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        output = model(input_ids=input_ids, word_spans=spans)
    index = max(0, len(encoded.word_spans) - 1)
    return output.logits[0, index]


def measure_mode(
    model,
    tokenizer: HybridTokenizer,
    model_type: str,
    device: torch.device,
    input_tokens: int,
    output_tokens: int,
    sampling: bool,
) -> float:
    if model_type == "tritmorph":
        effective_input = min(input_tokens, model.config.max_position_embeddings)
    else:
        effective_input = min(input_tokens, model.config.max_position_embeddings)
    seed_text = " ".join(["token"] * effective_input)
    current = seed_text

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(output_tokens):
        if model_type == "tritmorph":
            logits = next_logits_tritmorph(model, tokenizer, current, device)
        else:
            encoded = tokenizer.encode(current, add_special_tokens=False)
            token_ids = encoded.word_ids or [1]
            input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
            with torch.no_grad():
                output = model(input_ids=input_ids)
            logits = output.logits[0, -1]
        probs = torch.softmax(logits, dim=-1)
        if sampling:
            next_id = int(torch.multinomial(probs, num_samples=1).item())
        else:
            next_id = int(torch.argmax(probs).item())
        current = f"{current} w{next_id}"
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = max(time.perf_counter() - start, 1e-6)
    return output_tokens / elapsed


def build_markdown(model_type: str, checkpoint: Path, rows: list[SpeedRow], input_tokens: int, output_tokens: int) -> str:
    lines = [
        "# Speed Test Results",
        "",
        f"Model type: `{model_type}`",
        f"Checkpoint: `{checkpoint}`",
        f"Prompt setup: {input_tokens} input tokens -> {output_tokens} generated tokens.",
        "",
        "| Mode | Device | Ternary | Tokens/sec |",
        "| --- | --- | --- | ---: |",
    ]
    for row in rows:
        lines.append(f"| {row.mode} | {row.device} | {row.ternary} | {row.tokens_per_second:.2f} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    rows: list[SpeedRow] = []
    device_order = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    for device_name in device_order:
        device = torch.device(device_name)
        for ternary in [True, False]:
            model, tokenizer = build_model_from_checkpoint(args.checkpoint, args.model_type, device, force_ternary=ternary)
            greedy = measure_mode(
                model,
                tokenizer,
                args.model_type,
                device,
                args.input_tokens,
                args.output_tokens,
                sampling=False,
            )
            sampled = measure_mode(
                model,
                tokenizer,
                args.model_type,
                device,
                args.input_tokens,
                args.output_tokens,
                sampling=True,
            )
            rows.append(SpeedRow("greedy", device_name, str(ternary).lower(), greedy))
            rows.append(SpeedRow("sampling", device_name, str(ternary).lower(), sampled))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    markdown = build_markdown(args.model_type, args.checkpoint, rows, args.input_tokens, args.output_tokens)
    RESULTS_PATH.write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
