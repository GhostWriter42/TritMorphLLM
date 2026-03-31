"""Data utilities for TritMorphLLM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Iterator, List

from datasets import Dataset, load_dataset


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
    },
    "tiny_stories": {
        "name": "TinyStories",
        "config": None,
        "text_column": "text",
        "split_train": "train",
        "split_val": "validation",
    },
}


def resolve_dataset_config(config: dict[str, Any], dataset_name: str | None = None) -> dict[str, Any]:
    dataset_cfg = dict(config["dataset"])
    preset = dataset_name or dataset_cfg.get("preset")
    if preset in DATASET_PRESETS:
        resolved = dict(DATASET_PRESETS[preset])
        resolved["preset"] = preset
        return resolved
    return dataset_cfg


def load_text_splits(config: dict[str, Any], dataset_name: str | None = None) -> tuple[Dataset, Dataset, dict[str, Any]]:
    dataset_cfg = resolve_dataset_config(config, dataset_name=dataset_name)
    train = load_dataset(dataset_cfg["name"], dataset_cfg["config"], split=dataset_cfg["split_train"])
    val = load_dataset(dataset_cfg["name"], dataset_cfg["config"], split=dataset_cfg["split_val"])
    return train, val, dataset_cfg
