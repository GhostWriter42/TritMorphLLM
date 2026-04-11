"""Data utilities for TritMorphLLM."""

from __future__ import annotations

import random
from dataclasses import dataclass
from itertools import cycle
from typing import Any, Iterable, Iterator, List

from datasets import Dataset, IterableDataset, load_dataset


@dataclass(slots=True)
class TextBatch:
    """Simple text batch wrapper used by small experiments."""

    texts: List[str]


def batched(iterable: Iterable[str], batch_size: int) -> Iterator[TextBatch]:
    """Yield text items in fixed-size batches."""

    bucket: List[str] = []
    for item in iterable:
        bucket.append(item)
        if len(bucket) == batch_size:
            yield TextBatch(texts=bucket)
            bucket = []
    if bucket:
        yield TextBatch(texts=bucket)


DATASET_PRESETS: dict[str, dict[str, Any]] = {
    "wikitext103": {
        "name": "wikitext",
        "config": "wikitext-103-raw-v1",
        "text_column": "text",
        "split_train": "train",
        "split_val": "validation",
        "streaming": False,
    },
    "tiny_stories": {
        "name": "TinyStories",
        "config": None,
        "text_column": "text",
        "split_train": "train",
        "split_val": "validation",
        "streaming": False,
    },
    "fineweb_edu_code_agentic_mix": {
        "preset": "fineweb_edu_code_agentic_mix",
        "text_column": "text",
        "streaming": True,
        "split_train": "train",
        "split_val": "train",
        "sources": {
            "fineweb_edu": {
                "name": "HuggingFaceFW/fineweb-edu",
                "config": "sample-10BT",
                "split": "train",
                "text_field": "text",
            },
            "code": {
                "name": "bigcode/the-stack-dedup",
                "config": "python",
                "split": "train",
                "text_field": "content",
                "fallback_name": "wikitext",
                "fallback_config": "wikitext-103-raw-v1",
                "fallback_text_field": "text",
            },
            "agentic": {
                "name": "teknium/OpenHermes-2.5",
                "config": None,
                "split": "train",
                "text_field": "text",
            },
        },
    },
}


def resolve_dataset_config(config: dict[str, Any], dataset_name: str | None = None) -> dict[str, Any]:
    dataset_cfg = dict(config["dataset"])
    preset = dataset_name or dataset_cfg.get("preset")
    if preset in DATASET_PRESETS:
        resolved = dict(DATASET_PRESETS[preset])
        resolved["preset"] = preset
        if "textmix" in dataset_cfg:
            resolved["textmix"] = dict(dataset_cfg["textmix"])
        return resolved
    return dataset_cfg


def _normalize_stream_row(row: dict[str, Any], text_field: str) -> str:
    value = row.get(text_field, "")
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "\n".join(str(item) for item in value if item)
    if isinstance(value, dict):
        if "messages" in value and isinstance(value["messages"], list):
            return "\n".join(str(message.get("content", "")) for message in value["messages"] if isinstance(message, dict))
        return str(value)
    return str(value).strip()


def _iter_weighted_mix(config: dict[str, Any], limit: int | None = None) -> Iterator[str]:
    sources = config["sources"]
    mix = config.get("textmix", {"fineweb_edu": 0.70, "code": 0.20, "agentic": 0.10})
    datasets: dict[str, Iterator[dict[str, Any]]] = {}
    for name, source_cfg in sources.items():
        try:
            dataset = load_dataset(
                source_cfg["name"],
                source_cfg["config"],
                split=source_cfg["split"],
                streaming=True,
            )
        except Exception:
            fallback_name = source_cfg.get("fallback_name")
            if fallback_name is None:
                raise
            dataset = load_dataset(
                fallback_name,
                source_cfg.get("fallback_config"),
                split=source_cfg["split"],
                streaming=True,
            )
            if "fallback_text_field" in source_cfg:
                source_cfg["text_field"] = source_cfg["fallback_text_field"]
        datasets[name] = iter(dataset)

    population = list(mix.keys())
    weights = [float(mix[name]) for name in population]
    rng = random.Random(42)
    emitted = 0
    while limit is None or emitted < limit:
        source_name = rng.choices(population=population, weights=weights, k=1)[0]
        source_cfg = sources[source_name]
        row = next(datasets[source_name])
        text = _normalize_stream_row(row, source_cfg["text_field"])
        if text:
            emitted += 1
            yield text


def load_text_splits(config: dict[str, Any], dataset_name: str | None = None) -> tuple[Any, Any, dict[str, Any]]:
    dataset_cfg = resolve_dataset_config(config, dataset_name=dataset_name)
    if dataset_cfg.get("preset") == "fineweb_edu_code_agentic_mix":
        train_stream = _iter_weighted_mix(dataset_cfg, limit=config["training"]["max_train_samples"])
        val_stream = _iter_weighted_mix(dataset_cfg, limit=config["training"]["max_eval_samples"])
        return train_stream, val_stream, dataset_cfg

    train = load_dataset(dataset_cfg["name"], dataset_cfg["config"], split=dataset_cfg["split_train"])
    val = load_dataset(dataset_cfg["name"], dataset_cfg["config"], split=dataset_cfg["split_val"])
    return train, val, dataset_cfg
