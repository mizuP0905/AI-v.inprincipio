import math
from typing import Optional

import torch
from torch import nn


class SynapseGraph(nn.Module):
    def __init__(self, src_idx: torch.Tensor, P_target: int, K: int) -> None:
        super().__init__()
        self.P_target = P_target
        self.K = K
        self.register_buffer("src_idx", src_idx.long())

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        # y: (B, T, D)
        B, T, D = y.shape
        y_flat = y.reshape(B * T, D)
        idx = self.src_idx.reshape(-1)
        gathered = y_flat.index_select(1, idx)
        return gathered.view(B, T, self.P_target, self.K)

    @staticmethod
    def generate_src_idx(
        positions_src: torch.Tensor,
        positions_tgt: torch.Tensor,
        alpha_source: int,
        K: int,
        lambda_local: float,
        rho_lr: float,
        r_hub: float,
        rho_hub: float,
        unique_per_target: bool,
        allow_self: bool,
        seed: int,
        same_layer: bool = False,
    ) -> torch.Tensor:
        P_src = positions_src.shape[0]
        P_tgt = positions_tgt.shape[0]
        D = P_src * alpha_source

        lambda_local = max(lambda_local, 1e-6)
        rho_lr = min(max(rho_lr, 0.0), 1.0)
        rho_hub = min(max(rho_hub, 0.0), 1.0)

        if r_hub <= 0.0:
            H = 1
        else:
            H = max(1, int(math.ceil(r_hub * P_src)))

        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed))

        hub_indices = torch.randperm(P_src, generator=gen)[:H]

        max_unique = D
        if same_layer and not allow_self:
            max_unique = max(0, D - alpha_source)
        enforce_unique = unique_per_target and (max_unique >= K)
        max_attempts = max(32, K * 2)

        src_idx = torch.empty((P_tgt, K), dtype=torch.long)

        for q in range(P_tgt):
            dist = torch.norm(positions_src - positions_tgt[q], dim=1)
            logits = -dist / lambda_local
            probs = torch.softmax(logits, dim=0)
            used = torch.zeros(D, dtype=torch.bool) if enforce_unique else None

            for k in range(K):
                d = SynapseGraph._sample_port_index(
                    q=q,
                    probs=probs,
                    P_src=P_src,
                    alpha_source=alpha_source,
                    rho_lr=rho_lr,
                    rho_hub=rho_hub,
                    hub_indices=hub_indices,
                    allow_self=allow_self,
                    same_layer=same_layer,
                    gen=gen,
                )

                if enforce_unique:
                    attempts = 0
                    while used[d] and attempts < max_attempts:
                        d = SynapseGraph._sample_port_index(
                            q=q,
                            probs=probs,
                            P_src=P_src,
                            alpha_source=alpha_source,
                            rho_lr=rho_lr,
                            rho_hub=rho_hub,
                            hub_indices=hub_indices,
                            allow_self=allow_self,
                            same_layer=same_layer,
                            gen=gen,
                        )
                        attempts += 1

                    if used[d]:
                        remaining = torch.nonzero(~used).flatten()
                        if remaining.numel() > 0:
                            r_idx = torch.randint(
                                0, remaining.numel(), (1,), generator=gen
                            ).item()
                            d = remaining[r_idx].item()

                    used[d] = True

                src_idx[q, k] = d

        return src_idx

    @staticmethod
    def _sample_port_index(
        q: int,
        probs: torch.Tensor,
        P_src: int,
        alpha_source: int,
        rho_lr: float,
        rho_hub: float,
        hub_indices: torch.Tensor,
        allow_self: bool,
        same_layer: bool,
        gen: torch.Generator,
    ) -> int:
        while True:
            u = torch.rand((), generator=gen).item()
            if u < rho_lr:
                v = torch.rand((), generator=gen).item()
                if v < rho_hub:
                    p_idx = torch.randint(
                        0, hub_indices.numel(), (1,), generator=gen
                    ).item()
                    p = int(hub_indices[p_idx].item())
                else:
                    p = int(torch.randint(0, P_src, (1,), generator=gen).item())
            else:
                p = int(torch.multinomial(probs, 1, generator=gen).item())

            if same_layer and not allow_self and p == q:
                continue

            g = int(torch.randint(0, alpha_source, (1,), generator=gen).item())
            return p * alpha_source + g
