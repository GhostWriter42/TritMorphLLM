#!/usr/bin/env python3
"""Unified training script for TritMorphLLM and Vanilla BPE baselines."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
import yaml
from datasets import Dataset
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset as TorchDataset
from tqdm.auto import tqdm

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from model.tritmorph_model import TritMorphConfig, TritMorphModel
from model.vanilla_bpe_baseline import VanillaBPEBaseline, VanillaBPEConfig, VanillaBPETokenizer
from tokenizer.hybrid_morph_bpe import HybridTokenizer, TokenizedExample, WordSpan
from utils.data import load_text_splits


class ExampleDataset(TorchDataset):
    def __init__(self, examples: list[dict[str, Tensor]]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        return self.examples[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TritMorphLLM or a Vanilla BPE baseline")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "default.yaml")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=["wikitext103", "tiny_stories", "fineweb_edu_code_agentic_mix"],
    )
    parser.add_argument("--model-type", type=str, default=None, choices=["tritmorph", "vanilla_bpe"])
    parser.add_argument("--save-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--use-ternary", nargs="?", const="true", default=None)
    return parser.parse_args()


def parse_optional_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Unable to parse boolean value: {value}")


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str | None) -> torch.device:
    if device_name is not None:
        return torch.device(device_name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def iter_texts(dataset: Dataset, text_column: str, limit: int | None = None) -> Iterator[str]:
    count = 0
    for row in dataset:
        if isinstance(row, str):
            text = row
        else:
            text = row.get(text_column, "")
        if isinstance(text, str) and text.strip():
            yield text
            count += 1
            if limit is not None and count >= limit:
                break


def normalize_word(word: str, lowercase: bool = True) -> str:
    return word.lower() if lowercase else word


def build_word_vocab(texts: Sequence[str], tokenizer: HybridTokenizer) -> dict[str, int]:
    vocab = {"[PAD]": 0, "[UNK]": 1}
    for text in texts:
        encoded = tokenizer.encode(text, add_special_tokens=False)
        for word in encoded.words:
            normalized = normalize_word(word, tokenizer.lowercase)
            if normalized not in vocab:
                vocab[normalized] = len(vocab)
    return vocab


def encode_target_words(words: Sequence[str], word_vocab: dict[str, int], lowercase: bool = True) -> list[int]:
    unk_id = word_vocab["[UNK]"]
    return [word_vocab.get(normalize_word(word, lowercase), unk_id) for word in words]


def trim_encoded_example(encoded: TokenizedExample, max_token_length: int) -> TokenizedExample | None:
    if len(encoded.input_ids) <= max_token_length:
        return encoded

    allowed_words: list[WordSpan] = []
    for span in encoded.word_spans:
        if span.end <= max_token_length:
            allowed_words.append(span)
        else:
            break
    if len(allowed_words) < 2:
        return None

    last_end = allowed_words[-1].end
    return TokenizedExample(
        input_ids=encoded.input_ids[:last_end],
        attention_mask=encoded.attention_mask[:last_end],
        word_spans=allowed_words,
        words=encoded.words[: len(allowed_words)],
        tokens=encoded.tokens[:last_end],
        word_ids=encoded.word_ids[: len(allowed_words)] if encoded.word_ids is not None else None,
    )


def build_tritmorph_examples(
    texts: Sequence[str],
    tokenizer: HybridTokenizer,
    word_vocab: dict[str, int],
    max_token_length: int,
) -> list[dict[str, Tensor]]:
    examples: list[dict[str, Tensor]] = []
    for text in texts:
        encoded = tokenizer.encode(text, add_special_tokens=False)
        encoded = trim_encoded_example(encoded, max_token_length)
        if encoded is None or len(encoded.word_spans) < 2:
            continue

        word_spans_tensor = torch.full((len(encoded.word_spans), 2), -1, dtype=torch.long)
        for index, span in enumerate(encoded.word_spans):
            word_spans_tensor[index, 0] = span.start
            word_spans_tensor[index, 1] = span.end

        word_ids = encode_target_words(encoded.words, word_vocab, tokenizer.lowercase)
        labels = torch.full((len(word_ids),), -100, dtype=torch.long)
        for idx in range(len(word_ids) - 1):
            labels[idx] = word_ids[idx + 1]

        examples.append(
            {
                "input_ids": torch.tensor(encoded.input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(encoded.attention_mask, dtype=torch.long),
                "word_spans": word_spans_tensor,
                "labels": labels,
            }
        )
    return examples


def build_baseline_examples(
    texts: Sequence[str],
    hybrid_tokenizer: HybridTokenizer,
    baseline_tokenizer: VanillaBPETokenizer,
    word_vocab: dict[str, int],
    max_words: int,
) -> list[dict[str, Tensor]]:
    examples: list[dict[str, Tensor]] = []
    for text in texts:
        encoded = hybrid_tokenizer.encode(text, add_special_tokens=False)
        words = encoded.words[:max_words]
        if len(words) < 2:
            continue
        input_ids = baseline_tokenizer.encode_words(list(words))
        labels = torch.full((len(input_ids),), -100, dtype=torch.long)
        target_ids = encode_target_words(words, word_vocab, hybrid_tokenizer.lowercase)
        for idx in range(len(target_ids) - 1):
            labels[idx] = target_ids[idx + 1]
        examples.append(
            {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": labels,
            }
        )
    return examples


def pad_tritmorph_batch(batch: Sequence[dict[str, Tensor]], pad_token_id: int) -> dict[str, Tensor]:
    batch_size = len(batch)
    max_tokens = max(item["input_ids"].size(0) for item in batch)
    max_words = max(item["word_spans"].size(0) for item in batch)
    input_ids = torch.full((batch_size, max_tokens), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_tokens), dtype=torch.long)
    word_spans = torch.full((batch_size, max_words, 2), -1, dtype=torch.long)
    labels = torch.full((batch_size, max_words), -100, dtype=torch.long)
    for index, item in enumerate(batch):
        token_count = item["input_ids"].size(0)
        word_count = item["word_spans"].size(0)
        label_count = item["labels"].size(0)
        input_ids[index, :token_count] = item["input_ids"]
        attention_mask[index, :token_count] = item["attention_mask"]
        word_spans[index, :word_count] = item["word_spans"]
        labels[index, :label_count] = item["labels"]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "word_spans": word_spans,
        "labels": labels,
    }


def pad_baseline_batch(batch: Sequence[dict[str, Tensor]], pad_token_id: int) -> dict[str, Tensor]:
    batch_size = len(batch)
    max_len = max(item["input_ids"].size(0) for item in batch)
    input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
    for index, item in enumerate(batch):
        token_count = item["input_ids"].size(0)
        label_count = item["labels"].size(0)
        input_ids[index, :token_count] = item["input_ids"]
        labels[index, :label_count] = item["labels"]
    return {"input_ids": input_ids, "labels": labels}


def prepare_training_components(
    config: dict[str, Any],
    model_type: str,
    dataset_name: str | None = None,
) -> tuple[Any, nn.Module, DataLoader, DataLoader, dict[str, int]]:
    train_split, val_split, dataset_cfg = load_text_splits(config, dataset_name=dataset_name)
    text_column = dataset_cfg["text_column"]
    training_cfg = config["training"]
    tokenizer_cfg = config["tokenizer"]

    train_texts = list(iter_texts(train_split, text_column, training_cfg["max_train_samples"]))
    val_texts = list(iter_texts(val_split, text_column, training_cfg["max_eval_samples"]))

    hybrid_tokenizer = HybridTokenizer(
        vocab_size=tokenizer_cfg["vocab_size"],
        lowercase=tokenizer_cfg["lowercase"],
        min_frequency=tokenizer_cfg["min_frequency"],
        max_word_length=tokenizer_cfg["max_word_length"],
    )
    hybrid_tokenizer.train_from_iterator(train_texts)
    word_vocab = build_word_vocab(train_texts, hybrid_tokenizer)
    word_vocab["[UNK]"] = 1
    word_vocab["[PAD]"] = 0

    if model_type == "tritmorph":
        train_examples = build_tritmorph_examples(train_texts, hybrid_tokenizer, word_vocab, tokenizer_cfg["max_token_length"])
        val_examples = build_tritmorph_examples(val_texts, hybrid_tokenizer, word_vocab, tokenizer_cfg["max_token_length"])
        train_loader = DataLoader(
            ExampleDataset(train_examples),
            batch_size=training_cfg["batch_size"],
            shuffle=True,
            num_workers=training_cfg["num_workers"],
            collate_fn=lambda batch: pad_tritmorph_batch(batch, hybrid_tokenizer.pad_token_id),
        )
        val_loader = DataLoader(
            ExampleDataset(val_examples),
            batch_size=training_cfg["eval_batch_size"],
            shuffle=False,
            num_workers=training_cfg["num_workers"],
            collate_fn=lambda batch: pad_tritmorph_batch(batch, hybrid_tokenizer.pad_token_id),
        )
        config["model"]["vocab_size"] = hybrid_tokenizer.vocab_size_actual
        config["model"]["word_vocab_size"] = len(word_vocab)
        model = TritMorphModel(TritMorphConfig(**config["model"]))
        return hybrid_tokenizer, model, train_loader, val_loader, word_vocab

    baseline_tokenizer = VanillaBPETokenizer(
        vocab_size=tokenizer_cfg["vocab_size"],
        min_frequency=tokenizer_cfg["min_frequency"],
        lowercase=tokenizer_cfg["lowercase"],
    )
    baseline_tokenizer.train_from_iterator(train_texts)
    max_words = config["model"]["max_position_embeddings"]
    train_examples = build_baseline_examples(train_texts, hybrid_tokenizer, baseline_tokenizer, word_vocab, max_words)
    val_examples = build_baseline_examples(val_texts, hybrid_tokenizer, baseline_tokenizer, word_vocab, max_words)
    train_loader = DataLoader(
        ExampleDataset(train_examples),
        batch_size=training_cfg["batch_size"],
        shuffle=True,
        num_workers=training_cfg["num_workers"],
        collate_fn=lambda batch: pad_baseline_batch(batch, baseline_tokenizer.pad_token_id),
    )
    val_loader = DataLoader(
        ExampleDataset(val_examples),
        batch_size=training_cfg["eval_batch_size"],
        shuffle=False,
        num_workers=training_cfg["num_workers"],
        collate_fn=lambda batch: pad_baseline_batch(batch, baseline_tokenizer.pad_token_id),
    )
    baseline_cfg = VanillaBPEConfig(
        vocab_size=baseline_tokenizer.vocab_size_actual,
        word_vocab_size=len(word_vocab),
        max_position_embeddings=config["model"]["max_position_embeddings"],
        d_model=config["model"]["d_model"],
        n_heads=config["model"]["n_heads"],
        n_layers=config["model"]["n_layers"],
        mlp_ratio=config["model"]["mlp_ratio"],
        dropout=config["model"]["dropout"],
        pad_token_id=baseline_tokenizer.pad_token_id,
        use_ternary=config["model"]["use_ternary"],
    )
    model = VanillaBPEBaseline(baseline_cfg)
    return baseline_tokenizer, model, train_loader, val_loader, word_vocab


def cosine_lr(step: int, total_steps: int, warmup_steps: int, base_lr: float) -> float:
    if step < warmup_steps:
        return base_lr * float(step + 1) / float(max(1, warmup_steps))
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def forward_batch(model_type: str, model: nn.Module, batch: dict[str, Tensor]) -> Any:
    if model_type == "tritmorph":
        return model(input_ids=batch["input_ids"], word_spans=batch["word_spans"], labels=batch["labels"])
    return model(input_ids=batch["input_ids"], labels=batch["labels"])


@torch.no_grad()
def evaluate(model_type: str, model: nn.Module, dataloader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    losses: list[float] = []
    for batch in dataloader:
        batch = {key: value.to(device) for key, value in batch.items()}
        output = forward_batch(model_type, model, batch)
        if output.loss is not None:
            losses.append(output.loss.item())
    mean_loss = float(np.mean(losses)) if losses else float("nan")
    perplexity = float(math.exp(mean_loss)) if math.isfinite(mean_loss) else float("inf")
    model.train()
    return mean_loss, perplexity


def maybe_init_wandb(config: dict[str, Any], model_type: str) -> None:
    if not config["training"]["use_wandb"]:
        return
    if wandb is None:
        raise RuntimeError("wandb is enabled in config but not installed.")
    wandb.init(
        project=config["training"]["project"],
        name=f"{config['training']['run_name']}-{model_type}",
        config=config,
    )


def save_checkpoint(
    model: nn.Module,
    optimizer: AdamW,
    tokenizer: Any,
    word_vocab: dict[str, int],
    output_dir: Path,
    step: int,
    config: dict[str, Any],
    model_type: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"{model_type}_step_{step}.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
            "step": step,
            "model_type": model_type,
        },
        checkpoint_path,
    )
    if hasattr(tokenizer, "save"):
        tokenizer.save(output_dir / f"{model_type}_tokenizer.json")
    (output_dir / f"{model_type}_word_vocab.json").write_text(json.dumps(word_vocab, indent=2), encoding="utf-8")


def load_training_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: AdamW,
    device: torch.device,
) -> tuple[nn.Module, AdamW, int, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    step = int(checkpoint.get("step", 0))
    saved_config = checkpoint.get("config", {})
    return model, optimizer, step, saved_config


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.max_steps is not None:
        config["training"]["max_steps"] = args.max_steps
    if args.model_type is not None:
        config["training"]["model_type"] = args.model_type
    output_dir_override = args.output_dir or args.save_dir
    if output_dir_override is not None:
        config["training"]["output_dir"] = str(output_dir_override)
    parsed_use_ternary = parse_optional_bool(args.use_ternary)
    if parsed_use_ternary is not None:
        config["model"]["use_ternary"] = parsed_use_ternary
    if args.dataset is not None:
        config["dataset"]["preset"] = args.dataset

    set_seed(config["seed"])
    device = resolve_device(args.device or config.get("device"))
    model_type = config["training"].get("model_type", "tritmorph")
    print(f"Using output directory: {config['training']['output_dir']}")
    print(f"Using ternary layers: {config['model']['use_ternary']}")
    print(f"Using dataset preset: {config['dataset'].get('preset', 'custom')}")
    if args.resume_from is not None:
        print(f"Resuming from checkpoint: {args.resume_from}")

    tokenizer, model, train_loader, val_loader, word_vocab = prepare_training_components(
        config,
        model_type,
        dataset_name=config["dataset"].get("preset"),
    )
    model = model.to(device)
    compile_enabled = bool(config["training"].get("compile", False)) and device.type == "cuda"
    if compile_enabled:
        model = torch.compile(model)

    optimizer = AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        betas=tuple(config["training"]["betas"]),
        weight_decay=config["training"]["weight_decay"],
    )

    maybe_init_wandb(config, model_type)
    output_dir = ROOT / config["training"]["output_dir"]
    total_steps = config["training"]["max_steps"]
    grad_accum_steps = config["training"]["grad_accum_steps"]
    global_step = 0
    optimizer.zero_grad(set_to_none=True)

    if args.resume_from is not None:
        model, optimizer, global_step, saved_config = load_training_checkpoint(args.resume_from, model, optimizer, device)
        if isinstance(saved_config, dict) and saved_config:
            config = saved_config
            total_steps = config["training"]["max_steps"]
        if global_step >= total_steps:
            print(f"Checkpoint already reached step {global_step}, skipping training.")
            save_checkpoint(model, optimizer, tokenizer, word_vocab, output_dir, global_step, config, model_type)
            if config["training"]["use_wandb"] and wandb is not None:
                wandb.finish()
            return

    progress = tqdm(total=total_steps, desc=f"training:{model_type}")
    if global_step > 0:
        progress.update(global_step)

    model.train()
    while global_step < total_steps:
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            lr = cosine_lr(global_step, total_steps, config["training"]["warmup_steps"], config["training"]["learning_rate"])
            for group in optimizer.param_groups:
                group["lr"] = lr

            output = forward_batch(model_type, model, batch)
            if output.loss is None:
                continue

            (output.loss / grad_accum_steps).backward()
            should_step = ((global_step + 1) % grad_accum_steps == 0) or (global_step + 1 == total_steps)
            if should_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            global_step += 1
            progress.update(1)

            if global_step % config["training"]["log_interval"] == 0:
                train_loss = float(output.loss.item())
                train_ppl = math.exp(min(train_loss, 20.0))
                tqdm.write(f"step={global_step} loss={train_loss:.4f} ppl={train_ppl:.2f} lr={lr:.6f}")
                if config["training"]["use_wandb"] and wandb is not None:
                    wandb.log({"train/loss": train_loss, "train/ppl": train_ppl, "lr": lr}, step=global_step)

            if global_step % config["training"]["eval_interval"] == 0:
                val_loss, val_ppl = evaluate(model_type, model, val_loader, device)
                tqdm.write(f"eval step={global_step} loss={val_loss:.4f} ppl={val_ppl:.2f}")
                if config["training"]["use_wandb"] and wandb is not None:
                    wandb.log({"eval/loss": val_loss, "eval/ppl": val_ppl}, step=global_step)

            if global_step % config["training"]["save_interval"] == 0:
                save_checkpoint(model, optimizer, tokenizer, word_vocab, output_dir, global_step, config, model_type)

            if global_step >= total_steps:
                break

    save_checkpoint(model, optimizer, tokenizer, word_vocab, output_dir, global_step, config, model_type)
    if config["training"]["use_wandb"] and wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
