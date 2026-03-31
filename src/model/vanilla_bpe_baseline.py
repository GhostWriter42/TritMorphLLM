"""Vanilla BPE baseline without morphology-aware composition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.trainers import BpeTrainer

from .ternary_layers import RMSNorm, build_linear


SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]


class VanillaBPETokenizer:
    def __init__(self, vocab_size: int = 8192, min_frequency: int = 2, lowercase: bool = True) -> None:
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.lowercase = lowercase
        self.tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        self.tokenizer.pre_tokenizer = WhitespaceSplit()
        self._trained = False

    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.token_to_id("[PAD]") or 0

    @property
    def unk_token_id(self) -> int:
        return self.tokenizer.token_to_id("[UNK]") or 1

    @property
    def vocab_size_actual(self) -> int:
        return self.tokenizer.get_vocab_size()

    def train_from_iterator(self, texts: list[str]) -> None:
        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=self.min_frequency,
            special_tokens=SPECIAL_TOKENS,
            continuing_subword_prefix="##",
        )
        corpus = [text.lower() if self.lowercase else text for text in texts]
        self.tokenizer.train_from_iterator(corpus, trainer=trainer)
        self._trained = True

    def encode_words(self, words: list[str]) -> list[int]:
        ids: list[int] = []
        for word in words:
            normalized = word.lower() if self.lowercase else word
            token_id = self.tokenizer.token_to_id(normalized)
            ids.append(self.unk_token_id if token_id is None else token_id)
        return ids


@dataclass(slots=True)
class VanillaBPEConfig:
    vocab_size: int
    word_vocab_size: int
    max_position_embeddings: int = 256
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    mlp_ratio: int = 4
    dropout: float = 0.1
    pad_token_id: int = 0
    use_ternary: bool = False


@dataclass(slots=True)
class VanillaBPEOutput:
    logits: Tensor
    loss: Optional[Tensor]


class BaselineAttention(nn.Module):
    def __init__(self, config: VanillaBPEConfig) -> None:
        super().__init__()
        if config.d_model % config.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.qkv = build_linear(config.d_model, config.d_model * 3, bias=True, use_ternary=config.use_ternary)
        self.out_proj = build_linear(config.d_model, config.d_model, bias=True, use_ternary=config.use_ternary)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        batch_size, seq_len, hidden_dim = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        attn_scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        causal_mask = torch.triu(torch.ones((seq_len, seq_len), device=x.device, dtype=torch.bool), diagonal=1)
        attn_scores = attn_scores.masked_fill(causal_mask, float("-inf"))
        if attention_mask is not None:
            key_mask = ~attention_mask[:, None, None, :]
            attn_scores = attn_scores.masked_fill(key_mask, float("-inf"))
        probs = torch.softmax(attn_scores, dim=-1)
        probs = self.dropout(probs)
        output = probs @ v
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_dim)
        return self.out_proj(output)


class BaselineBlock(nn.Module):
    def __init__(self, config: VanillaBPEConfig) -> None:
        super().__init__()
        self.ln_1 = RMSNorm(config.d_model) if config.use_ternary else nn.LayerNorm(config.d_model)
        self.attn = BaselineAttention(config)
        self.ln_2 = RMSNorm(config.d_model) if config.use_ternary else nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(
            build_linear(config.d_model, config.d_model * config.mlp_ratio, bias=True, use_ternary=config.use_ternary),
            nn.GELU(),
            build_linear(config.d_model * config.mlp_ratio, config.d_model, bias=True, use_ternary=config.use_ternary),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: Tensor, attention_mask: Optional[Tensor]) -> Tensor:
        x = x + self.attn(self.ln_1(x), attention_mask=attention_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class VanillaBPEBaseline(nn.Module):
    def __init__(self, config: VanillaBPEConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.word_vocab_size, config.d_model, padding_idx=config.pad_token_id)
        self.position_embedding = nn.Embedding(config.max_position_embeddings, config.d_model)
        self.blocks = nn.ModuleList([BaselineBlock(config) for _ in range(config.n_layers)])
        self.final_norm = RMSNorm(config.d_model) if config.use_ternary else nn.LayerNorm(config.d_model)
        self.lm_head = build_linear(config.d_model, config.word_vocab_size, bias=False, use_ternary=config.use_ternary)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids: Tensor, labels: Optional[Tensor] = None) -> VanillaBPEOutput:
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        attention_mask = input_ids.ne(self.config.pad_token_id)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x, attention_mask=attention_mask)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
                labels[:, 1:].contiguous().view(-1),
                ignore_index=-100,
            )
        return VanillaBPEOutput(logits=logits, loss=loss)
