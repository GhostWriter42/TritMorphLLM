"""TritMorph language model with explicit composition before the transformer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .composition_layer import CompositionLayer, CompositionOutput
from .ternary_layers import RMSNorm, build_linear


@dataclass(slots=True)
class TritMorphConfig:
    vocab_size: int
    max_position_embeddings: int = 256
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    mlp_ratio: int = 4
    dropout: float = 0.1
    composition_hidden_dim: int = 512
    composition_attention: bool = True
    pad_token_id: int = 0
    word_vocab_size: int = 8192
    use_ternary: bool = False


@dataclass(slots=True)
class TritMorphOutput:
    logits: Tensor
    loss: Optional[Tensor]
    composition: CompositionOutput


class CausalSelfAttention(nn.Module):
    def __init__(self, config: TritMorphConfig) -> None:
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
        causal_mask = torch.triu(
            torch.ones((seq_len, seq_len), device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        attn_scores = attn_scores.masked_fill(causal_mask, float("-inf"))

        if attention_mask is not None:
            key_mask = ~attention_mask[:, None, None, :]
            attn_scores = attn_scores.masked_fill(key_mask, float("-inf"))

        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        output = attn_probs @ v
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_dim)
        return self.out_proj(output)


class TransformerBlock(nn.Module):
    def __init__(self, config: TritMorphConfig) -> None:
        super().__init__()
        self.ln_1 = RMSNorm(config.d_model) if config.use_ternary else nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.d_model) if config.use_ternary else nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(
            build_linear(
                config.d_model,
                config.d_model * config.mlp_ratio,
                bias=True,
                use_ternary=config.use_ternary,
            ),
            nn.GELU(),
            build_linear(
                config.d_model * config.mlp_ratio,
                config.d_model,
                bias=True,
                use_ternary=config.use_ternary,
            ),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        x = x + self.attn(self.ln_1(x), attention_mask=attention_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class TritMorphModel(nn.Module):
    """Small language model with explicit word composition after embeddings."""

    def __init__(self, config: TritMorphConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_token_id)
        self.position_embedding = nn.Embedding(config.max_position_embeddings, config.d_model)
        self.composition = CompositionLayer(
            hidden_dim=config.d_model,
            composition_hidden_dim=config.composition_hidden_dim,
            use_attention=config.composition_attention,
            dropout=config.dropout,
            use_ternary=config.use_ternary,
        )
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.final_norm = RMSNorm(config.d_model) if config.use_ternary else nn.LayerNorm(config.d_model)
        self.lm_head = build_linear(config.d_model, config.word_vocab_size, bias=False, use_ternary=config.use_ternary)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        input_ids: Tensor,
        word_spans: Tensor,
        labels: Optional[Tensor] = None,
    ) -> TritMorphOutput:
        token_embeddings = self.token_embedding(input_ids)
        composition = self.composition(token_embeddings, word_spans)

        batch_size, word_seq_len, _ = composition.word_embeddings.shape
        positions = torch.arange(word_seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        hidden_states = composition.word_embeddings + self.position_embedding(positions)
        hidden_states = self.dropout(hidden_states)

        attention_mask = composition.word_attention_mask
        for block in self.blocks:
            hidden_states = block(hidden_states, attention_mask=attention_mask)

        hidden_states = self.final_norm(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
                labels[:, 1:].contiguous().view(-1),
                ignore_index=-100,
            )

        return TritMorphOutput(logits=logits, loss=loss, composition=composition)
