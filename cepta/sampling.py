from typing import Optional

import torch


def sample_logits(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    if temperature <= 0.0:
        return torch.argmax(logits, dim=-1)

    logits = logits / temperature

    if top_k > 0:
        k = min(top_k, logits.size(-1))
        values, indices = torch.topk(logits, k, dim=-1)
        masked = torch.full_like(logits, float("-inf"))
        masked.scatter_(1, indices, values)
        logits = masked

    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = probs.cumsum(dim=-1)
        cutoff = cumulative > top_p
        cutoff[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(cutoff, float("-inf"))
        masked = torch.full_like(logits, float("-inf"))
        masked.scatter_(1, sorted_idx, sorted_logits)
        logits = masked

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1, generator=generator).squeeze(1)
