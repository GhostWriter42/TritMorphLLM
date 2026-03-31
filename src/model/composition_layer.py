"""Composition layer that fuses subwords into word-level representations."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .ternary_layers import build_linear


@dataclass(slots=True)
class CompositionOutput:
    """Outputs from the composition layer."""

    word_embeddings: Tensor
    word_attention_mask: Tensor
    word_to_subword_index: Tensor


class CompositionLayer(nn.Module):
    """Learned subword-to-word composition module.

    Each word span is fused with a gated weighted average and a small MLP. The
    layer is intentionally lightweight so it can be replaced by richer routing or
    ternary-friendly blocks in later phases.
    """

    def __init__(
        self,
        hidden_dim: int,
        composition_hidden_dim: int,
        use_attention: bool = True,
        dropout: float = 0.1,
        use_ternary: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_attention = use_attention

        self.score = build_linear(hidden_dim, 1, bias=True, use_ternary=use_ternary) if use_attention else None
        self.gate = nn.Sequential(
            build_linear(hidden_dim * 2, hidden_dim, bias=True, use_ternary=use_ternary),
            nn.Sigmoid(),
        )
        self.mlp = nn.Sequential(
            build_linear(hidden_dim, composition_hidden_dim, bias=True, use_ternary=use_ternary),
            nn.GELU(),
            nn.Dropout(dropout),
            build_linear(composition_hidden_dim, hidden_dim, bias=True, use_ternary=use_ternary),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, subword_embeddings: Tensor, word_spans: Tensor) -> CompositionOutput:
        """Compose subword embeddings into word-level embeddings.

        Args:
            subword_embeddings: Tensor of shape [batch, seq_len, hidden_dim].
            word_spans: Tensor of shape [batch, max_words, 2] with [start, end)
                token spans. Invalid spans should be filled with -1.
        """

        batch_size, _, hidden_dim = subword_embeddings.shape
        _, max_words, _ = word_spans.shape

        composed = subword_embeddings.new_zeros((batch_size, max_words, hidden_dim))
        mask = torch.zeros((batch_size, max_words), dtype=torch.bool, device=subword_embeddings.device)
        index = torch.full((batch_size, max_words), -1, dtype=torch.long, device=subword_embeddings.device)

        for batch_idx in range(batch_size):
            for word_idx in range(max_words):
                start = int(word_spans[batch_idx, word_idx, 0].item())
                end = int(word_spans[batch_idx, word_idx, 1].item())
                if start < 0 or end <= start:
                    continue

                pieces = subword_embeddings[batch_idx, start:end]
                fused = self._compose_word(pieces)
                composed[batch_idx, word_idx] = fused
                mask[batch_idx, word_idx] = True
                index[batch_idx, word_idx] = start

        return CompositionOutput(
            word_embeddings=composed,
            word_attention_mask=mask,
            word_to_subword_index=index,
        )

    def _compose_word(self, pieces: Tensor) -> Tensor:
        if pieces.size(0) == 1:
            single = pieces[0]
            return self.norm(single + self.dropout(self.mlp(single)))

        pooled = pieces.mean(dim=0)
        if self.use_attention and self.score is not None:
            scores = self.score(pieces).squeeze(-1)
            weights = torch.softmax(scores, dim=0)
            attended = torch.sum(pieces * weights.unsqueeze(-1), dim=0)
        else:
            attended = pooled

        gate_input = torch.cat([pooled, attended], dim=-1)
        gate = self.gate(gate_input)
        fused = gate * attended + (1.0 - gate) * pooled
        fused = fused + self.dropout(self.mlp(fused))
        return self.norm(fused)
