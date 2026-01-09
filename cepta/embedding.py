import math
from typing import Optional, Tuple, Union

import torch
from torch import nn

from .ops import flatten_ports


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 2048) -> None:
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        self.register_buffer("pos_cache", self._build_cache(max_len), persistent=True)

    def _build_cache(self, length: int) -> torch.Tensor:
        position = torch.arange(0, length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.dim, 2, dtype=torch.float32)
            * (-math.log(10000.0) / self.dim)
        )
        pe = torch.zeros(length, self.dim, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def _ensure_length(self, needed: int) -> None:
        if needed <= self.pos_cache.shape[0]:
            return
        new_cache = self._build_cache(needed)
        self.pos_cache = new_cache.to(self.pos_cache.device)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        B, T, D = x.shape
        needed = offset + T
        self._ensure_length(needed)
        pos = self.pos_cache[offset:offset + T].unsqueeze(0).to(x.dtype)
        return x + pos


class CeptaEmbedding(nn.Module):
    def __init__(self, vocab_size: int, P: int, alpha: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.P = P
        self.alpha = alpha
        self.D = P * alpha
        self.embedding = nn.Embedding(vocab_size, self.D)

    def forward(
        self, input_ids: torch.Tensor, return_ports: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x = self.embedding(input_ids)
        y = x.reshape(x.shape[0], x.shape[1], self.P, self.alpha)
        x_flat = flatten_ports(y)
        if return_ports:
            return x_flat, y
        return x_flat
