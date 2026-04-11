"""lm-evaluation-harness wrapper for TritMorphLLM models."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model

from model.tritmorph_model import TritMorphConfig, TritMorphModel
from model.vanilla_bpe_baseline import VanillaBPEBaseline, VanillaBPEConfig
from tokenizer.hybrid_morph_bpe import HybridTokenizer


def _split_words(text: str) -> list[str]:
    return [token for token in text.replace("\n", " ").split(" ") if token]


@register_model("tritmorph")
class TritMorphHarnessLM(LM):
    """Custom LM wrapper that scores/generates at word level."""

    def __init__(self, checkpoint: str, model_type: str = "tritmorph", **kwargs: Any) -> None:
        super().__init__()
        self.checkpoint_path = Path(checkpoint)
        self.model_type = model_type
        device = str(kwargs.pop("device", "cuda"))
        batch_size = int(kwargs.pop("batch_size", 1))
        max_gen_toks = int(kwargs.pop("max_gen_toks", 64))
        self._device = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
        self._batch_size = batch_size
        self._max_gen_toks = max_gen_toks

        payload = torch.load(self.checkpoint_path, map_location=self._device)
        self.config: dict[str, Any] = payload["config"]
        self.step = int(payload.get("step", 0))
        self.output_dir = self.checkpoint_path.parent

        word_vocab_path = self.output_dir / f"{self.model_type}_word_vocab.json"
        if not word_vocab_path.exists():
            fallback = self.output_dir / "tritmorph_word_vocab.json"
            word_vocab_path = fallback if fallback.exists() else word_vocab_path
        self.word_vocab = json.loads(word_vocab_path.read_text(encoding="utf-8"))
        self.word_unk_id = int(self.word_vocab.get("[UNK]", 1))
        self.word_pad_id = int(self.word_vocab.get("[PAD]", 0))
        self.id_to_word = {int(idx): token for token, idx in self.word_vocab.items()}

        tokenizer_path = self.output_dir / f"{self.model_type}_tokenizer.json"
        if not tokenizer_path.exists():
            fallback_tok = self.output_dir / "tritmorph_tokenizer.json"
            tokenizer_path = fallback_tok if fallback_tok.exists() else tokenizer_path
        self.hybrid_tokenizer = HybridTokenizer.from_file(tokenizer_path)

        model_cfg = dict(self.config["model"])
        model_cfg["word_vocab_size"] = len(self.word_vocab)
        if self.model_type == "tritmorph":
            self.model = TritMorphModel(TritMorphConfig(**model_cfg)).to(self._device)
        else:
            baseline_cfg = VanillaBPEConfig(
                vocab_size=model_cfg["vocab_size"],
                word_vocab_size=len(self.word_vocab),
                max_position_embeddings=model_cfg["max_position_embeddings"],
                d_model=model_cfg["d_model"],
                n_heads=model_cfg["n_heads"],
                n_layers=model_cfg["n_layers"],
                mlp_ratio=model_cfg["mlp_ratio"],
                dropout=model_cfg["dropout"],
                pad_token_id=model_cfg["pad_token_id"],
                use_ternary=model_cfg.get("use_ternary", False),
            )
            self.model = VanillaBPEBaseline(baseline_cfg).to(self._device)

        self.model.load_state_dict(payload["model"], strict=False)
        self.model.eval()

    @property
    def eot_token_id(self) -> int:
        return self.word_pad_id

    @property
    def max_length(self) -> int:
        return int(self.config["model"].get("max_position_embeddings", 256))

    @property
    def max_gen_toks(self) -> int:
        return self._max_gen_toks

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def rank(self) -> int:
        return 0

    @property
    def world_size(self) -> int:
        return 1

    def tok_encode(self, string: str, **_: Any) -> list[int]:
        words = _split_words(string.lower())
        return [int(self.word_vocab.get(word, self.word_unk_id)) for word in words]

    def tok_decode(self, tokens: Iterable[int], **_: Any) -> str:
        return " ".join(self.id_to_word.get(int(token), "[UNK]") for token in tokens)

    def _build_tritmorph_inputs(self, text: str) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
        words = _split_words(text)
        if len(words) > self.max_length:
            text = " ".join(words[-self.max_length :])
        encoded = self.hybrid_tokenizer.encode(text, add_special_tokens=False)
        if not encoded.input_ids:
            encoded = self.hybrid_tokenizer.encode("[UNK]", add_special_tokens=False)
        if len(encoded.word_spans) > self.max_length:
            encoded.word_spans = encoded.word_spans[-self.max_length :]
            start = encoded.word_spans[0].start
            encoded.input_ids = encoded.input_ids[start:]
            encoded.word_spans = [
                type(span)(word=span.word, start=span.start - start, end=span.end - start)
                for span in encoded.word_spans
            ]
        spans = torch.full((1, max(1, len(encoded.word_spans)), 2), -1, dtype=torch.long, device=self._device)
        for idx, span in enumerate(encoded.word_spans):
            spans[0, idx, 0] = span.start
            spans[0, idx, 1] = span.end
        input_ids = torch.tensor([encoded.input_ids], dtype=torch.long, device=self._device)
        span_words = [span.word for span in encoded.word_spans]
        return input_ids, spans, span_words

    def _next_word_distribution(self, text: str) -> torch.Tensor:
        words = _split_words(text)
        if len(words) > self.max_length:
            text = " ".join(words[-self.max_length :])
        with torch.no_grad():
            if self.model_type == "tritmorph":
                input_ids, spans, words = self._build_tritmorph_inputs(text)
                output = self.model(input_ids=input_ids, word_spans=spans)
                valid_words = int(output.composition.word_attention_mask[0].sum().item())
                last_index = max(0, min(valid_words - 1, output.logits.size(1) - 1))
                logits = output.logits[0, last_index]
            else:
                tokens = self.tok_encode(text)
                if not tokens:
                    tokens = [self.word_unk_id]
                if len(tokens) > self.max_length:
                    tokens = tokens[-self.max_length :]
                input_ids = torch.tensor([tokens], dtype=torch.long, device=self._device)
                output = self.model(input_ids=input_ids)
                logits = output.logits[0, -1]
            return F.log_softmax(logits, dim=-1)

    def _score_continuation(self, context: str, continuation: str) -> tuple[float, bool]:
        cont_words = _split_words(continuation.lower())
        if not cont_words:
            return 0.0, True
        running_text = context.strip()
        total_logprob = 0.0
        greedy_match = True
        for word in cont_words:
            log_probs = self._next_word_distribution(running_text)
            token_id = int(self.word_vocab.get(word, self.word_unk_id))
            total_logprob += float(log_probs[token_id].item())
            pred_id = int(torch.argmax(log_probs).item())
            greedy_match = greedy_match and (pred_id == token_id)
            running_text = f"{running_text} {word}".strip()
        return total_logprob, greedy_match

    def loglikelihood(self, requests: list[Instance], disable_tqdm: bool = False) -> list[tuple[float, bool]]:
        del disable_tqdm
        outputs: list[tuple[float, bool]] = []
        for instance in requests:
            context, continuation = instance.arguments
            outputs.append(self._score_continuation(str(context), str(continuation)))
        return outputs

    def loglikelihood_rolling(self, requests: list[Instance], disable_tqdm: bool = False) -> list[float]:
        del disable_tqdm
        outputs: list[float] = []
        for instance in requests:
            text = str(instance.arguments[0])
            words = _split_words(text.lower())
            if len(words) < 2:
                outputs.append(0.0)
                continue
            total = 0.0
            for idx in range(1, len(words)):
                context = " ".join(words[:idx])
                continuation = words[idx]
                score, _ = self._score_continuation(context, continuation)
                total += score
            outputs.append(total)
        return outputs

    def generate_until(self, requests: list[Instance], disable_tqdm: bool = False) -> list[str]:
        del disable_tqdm
        outputs: list[str] = []
        for instance in requests:
            context = str(instance.arguments[0])
            gen_kwargs = instance.arguments[1] if len(instance.arguments) > 1 else {}
            stop_sequences = gen_kwargs.get("until", []) if isinstance(gen_kwargs, dict) else []
            max_tokens = int(gen_kwargs.get("max_gen_toks", self.max_gen_toks)) if isinstance(gen_kwargs, dict) else self.max_gen_toks
            current = context.strip()
            generated_words: list[str] = []
            for _ in range(max_tokens):
                log_probs = self._next_word_distribution(current)
                next_id = int(torch.argmax(log_probs).item())
                next_word = self.id_to_word.get(next_id, "[UNK]")
                generated_words.append(next_word)
                current = f"{current} {next_word}".strip()
                if any(stop in current for stop in stop_sequences):
                    break
            outputs.append(" ".join(generated_words))
        return outputs
