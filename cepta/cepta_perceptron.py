from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


class CeptaPerceptronDense(nn.Module):
    def __init__(
        self,
        P: int,
        K: int,
        alpha: int,
        use_ste: bool = False,
        ste_mode: str = "A",
        ste_tau: float = 1.0,
        dale_mode: bool = False,
        e_ratio: float = 0.8,
        neuron_types: Optional[torch.Tensor] = None,
        neuron_type_seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.P = P
        self.K = K
        self.alpha = alpha
        self.use_ste = use_ste
        self.ste_mode = ste_mode
        self.ste_tau = ste_tau
        self.dale_mode = dale_mode

        self.w = nn.Parameter(torch.empty(P, K))
        self.sp = nn.Parameter(torch.zeros(P))
        self.f_param = nn.Parameter(torch.empty(P, alpha))

        nn.init.xavier_uniform_(self.w)
        nn.init.xavier_uniform_(self.f_param)

        if self.dale_mode:
            if neuron_types is None:
                gen = torch.Generator(device="cpu")
                if neuron_type_seed is not None:
                    gen.manual_seed(int(neuron_type_seed))
                neuron_types = (torch.rand(P, generator=gen) < e_ratio).long()
            neuron_types = neuron_types.view(P, 1).long()
            self.register_buffer("neuron_types", neuron_types)
        else:
            self.register_buffer("neuron_types", torch.empty(0), persistent=False)

    def compute_f(self) -> torch.Tensor:
        if not self.dale_mode:
            return self.f_param.float()
        types = self.neuron_types.to(dtype=torch.float32, device=self.f_param.device)
        sign = torch.where(types > 0, 1.0, -1.0)
        return F.softplus(self.f_param.float()) * sign

    def forward(
        self, x_dend: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x_dend: (B, T, P, K)
        x_fp32 = x_dend.float()
        w_fp32 = self.w.float()
        sp_fp32 = self.sp.float()

        u = (x_fp32 * w_fp32).sum(dim=-1)
        f_hard = (u >= sp_fp32).float()

        if self.use_ste:
            if self.ste_mode == "A":
                f_gate = f_hard + (u - u.detach())
            elif self.ste_mode == "B":
                mask = (u - sp_fp32).abs() < self.ste_tau
                f_gate = f_hard + (u - u.detach()) * mask.float()
            else:
                raise ValueError(f"Unsupported STE mode: {self.ste_mode}")
        else:
            f_gate = f_hard

        f_out = self.compute_f()
        y = (f_gate * u).unsqueeze(-1) * f_out
        return u, f_hard, y

    def l2_stability_loss(self) -> torch.Tensor:
        # L2 stabilization for weights and thresholds.
        return 0.5 * (self.w.pow(2).mean() + self.sp.pow(2).mean())

    @torch.no_grad()
    def apply_oja_update(
        self,
        x_dend: torch.Tensor,
        activity: torch.Tensor,
        lr: float,
    ) -> None:
        # Oja update: w <- w + lr * (y x - y^2 w), averaged over (B, T).
        x_fp32 = x_dend.float()
        y_fp32 = activity.float()
        xy = (y_fp32.unsqueeze(-1) * x_fp32).mean(dim=(0, 1))
        y2 = (y_fp32.pow(2).mean(dim=(0, 1))).unsqueeze(-1)
        self.w.data.add_(lr * (xy - y2 * self.w.data))
