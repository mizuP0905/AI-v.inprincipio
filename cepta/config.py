from dataclasses import dataclass
from typing import Optional


@dataclass
class GraphConfig:
    P: int
    alpha: int
    K: int = 64
    lambda_local: float = 1.0
    rho_lr: float = 0.1
    r_hub: float = 0.1
    rho_hub: float = 0.5
    unique_per_target: bool = True
    allow_self: bool = False
    positions_mode: str = "random"  # "random" or "grid"


@dataclass
class SSMConfig:
    P: int
    P_r: int
    mode: str = "mode1"  # "mode1" or "mode2"
    a_min: float = 0.0
    a_max: float = 1.0
    rms_norm: bool = False
    rms_eps: float = 1e-5
    learnable_s0: bool = False


@dataclass
class PositionalConfig:
    mode: str = "A"  # "A" (additive) or "B" (none)
    max_seq_len: int = 2048


@dataclass
class CeptaConfig:
    vocab_size: int
    P: int
    alpha: int
    K: int = 64
    num_layers: int = 4
    use_ste: bool = False
    ste_mode: str = "A"  # "A" or "B"
    ste_tau: float = 1.0
    dale_mode: bool = False
    e_ratio: float = 0.8
    seed_base: int = 1234
    mlp_cross_layer: bool = True
    graph_ssm: Optional[GraphConfig] = None
    graph_mlp: Optional[GraphConfig] = None
    ssm: Optional[SSMConfig] = None
    pos: Optional[PositionalConfig] = None

    def build_graph_config(self) -> None:
        if self.graph_ssm is None:
            self.graph_ssm = GraphConfig(P=self.P, alpha=self.alpha, K=self.K)
        if self.graph_mlp is None:
            self.graph_mlp = GraphConfig(P=self.P, alpha=self.alpha, K=self.K)

    def build_ssm_config(self) -> None:
        if self.ssm is None:
            self.ssm = SSMConfig(P=self.P, P_r=max(1, self.P // 4))

    def build_pos_config(self) -> None:
        if self.pos is None:
            self.pos = PositionalConfig()

    def finalize(self) -> None:
        self.build_graph_config()
        self.build_ssm_config()
        self.build_pos_config()
