"""BitNet b1.58 style ternary layers for TritMorphLLM.

This implementation uses a straight-through estimator with absmax scaling and a
ternary value set {-1, 0, +1}. The forward pass keeps the core matmul path free
of dense learned floating-point weights by quantizing per output row on the fly.
The design is compact and intended to be a clean production-ready baseline for
later kernel specialization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def absmax_scale(weight: Tensor, dim: int = -1, eps: float = 1e-6) -> Tensor:
    """Compute absmax scaling factors."""

    return weight.abs().amax(dim=dim, keepdim=True).clamp_min(eps)


class _TernaryWeightFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight: Tensor, threshold: float) -> Tensor:
        del ctx
        scale = absmax_scale(weight, dim=-1)
        normalized = weight / scale
        ternary = torch.where(
            normalized > threshold,
            torch.ones_like(normalized),
            torch.where(normalized < -threshold, -torch.ones_like(normalized), torch.zeros_like(normalized)),
        )
        return ternary * scale

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, None]:
        del ctx
        return grad_output, None


def ternarize_weight(weight: Tensor, threshold: float = 0.5) -> Tensor:
    return _TernaryWeightFn.apply(weight, threshold)


class RMSNorm(nn.Module):
    """Simple RMSNorm used by ternary blocks."""

    def __init__(self, hidden_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return x * self.weight


class BitLinear(nn.Module):
    """BitNet-style ternary linear layer with STE quantization."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True, threshold: float = 0.5) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.threshold = threshold
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1.0 / fan_in ** 0.5
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        quantized_weight = ternarize_weight(self.weight, threshold=self.threshold)
        return F.linear(x, quantized_weight, self.bias)


def build_linear(in_features: int, out_features: int, bias: bool, use_ternary: bool) -> nn.Module:
    if use_ternary:
        return BitLinear(in_features, out_features, bias=bias)
    return nn.Linear(in_features, out_features, bias=bias)
