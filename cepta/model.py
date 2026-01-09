from typing import List, Optional, Tuple, Union

import torch
from torch import nn

from .block import CeptaBlock
from .config import CeptaConfig, GraphConfig, SSMConfig
from .embedding import CeptaEmbedding, PositionalEncoding
from .sampling import sample_logits
from .ssm import LowRankSSM
from .synapse_graph import SynapseGraph


class CeptaLM(nn.Module):
    def __init__(self, config: CeptaConfig) -> None:
        super().__init__()
        config.finalize()
        self.config = config

        if config.graph_ssm.positions_mode != config.graph_mlp.positions_mode:
            raise ValueError("positions_mode must be consistent across graphs.")

        self.P = config.P
        self.alpha = config.alpha
        self.D = config.P * config.alpha

        self.embedding = CeptaEmbedding(config.vocab_size, config.P, config.alpha)
        if config.pos.mode == "A":
            self.positional = PositionalEncoding(self.D, config.pos.max_seq_len)
        else:
            self.positional = None

        positions = self._build_positions(
            num_layers=config.num_layers + 1,
            P=config.P,
            mode=config.graph_ssm.positions_mode,
            seed_base=config.seed_base,
        )
        self.register_buffer("positions", positions)

        self.blocks = nn.ModuleList()
        for layer_idx in range(config.num_layers):
            graph_ssm = self._build_graph(
                layer_idx=layer_idx,
                role_id=1,
                graph_cfg=config.graph_ssm,
                cross_layer=True,
            )
            graph_mlp = self._build_graph(
                layer_idx=layer_idx,
                role_id=2,
                graph_cfg=config.graph_mlp,
                cross_layer=config.mlp_cross_layer,
            )
            ssm = self._build_ssm(config.ssm)
            neuron_type_seed = config.seed_base + 1000 * layer_idx + 30
            block = CeptaBlock(
                D=self.D,
                P=config.P,
                K=config.K,
                alpha=config.alpha,
                graph_ssm=graph_ssm,
                graph_mlp=graph_mlp,
                use_ste=config.use_ste,
                ste_mode=config.ste_mode,
                ste_tau=config.ste_tau,
                dale_mode=config.dale_mode,
                e_ratio=config.e_ratio,
                ssm=ssm,
                neuron_type_seed=neuron_type_seed,
            )
            self.blocks.append(block)

        self.lm_head = nn.Linear(self.D, config.vocab_size, bias=False)

    @staticmethod
    def _build_positions(
        num_layers: int, P: int, mode: str, seed_base: int
    ) -> torch.Tensor:
        positions = []
        for layer_idx in range(num_layers):
            seed = seed_base + 1000 * layer_idx + 90
            pos = CeptaLM._generate_positions(P, mode, seed)
            positions.append(pos)
        return torch.stack(positions, dim=0)

    @staticmethod
    def _generate_positions(P: int, mode: str, seed: int) -> torch.Tensor:
        if mode == "random":
            gen = torch.Generator(device="cpu")
            gen.manual_seed(int(seed))
            return torch.rand(P, 2, generator=gen, dtype=torch.float32)
        if mode == "grid":
            side = int(torch.ceil(torch.sqrt(torch.tensor(float(P))))).item()
            coords = torch.linspace(0.0, 1.0, steps=side, dtype=torch.float32)
            grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")
            grid = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)
            return grid[:P]
        raise ValueError(f"Unsupported positions mode: {mode}")

    def _build_graph(
        self,
        layer_idx: int,
        role_id: int,
        graph_cfg: GraphConfig,
        cross_layer: bool,
    ) -> SynapseGraph:
        if cross_layer:
            pos_src = self.positions[layer_idx]
            pos_tgt = self.positions[layer_idx + 1]
            same_layer = False
        else:
            pos_src = self.positions[layer_idx + 1]
            pos_tgt = self.positions[layer_idx + 1]
            same_layer = True

        seed = self.config.seed_base + 1000 * layer_idx + 10 * role_id
        src_idx = SynapseGraph.generate_src_idx(
            positions_src=pos_src,
            positions_tgt=pos_tgt,
            alpha_source=graph_cfg.alpha,
            K=graph_cfg.K,
            lambda_local=graph_cfg.lambda_local,
            rho_lr=graph_cfg.rho_lr,
            r_hub=graph_cfg.r_hub,
            rho_hub=graph_cfg.rho_hub,
            unique_per_target=graph_cfg.unique_per_target,
            allow_self=graph_cfg.allow_self,
            seed=seed,
            same_layer=same_layer,
        )
        return SynapseGraph(src_idx=src_idx, P_target=graph_cfg.P, K=graph_cfg.K)

    @staticmethod
    def _build_ssm(ssm_cfg: SSMConfig) -> LowRankSSM:
        return LowRankSSM(
            P=ssm_cfg.P,
            P_r=ssm_cfg.P_r,
            mode=ssm_cfg.mode,
            a_min=ssm_cfg.a_min,
            a_max=ssm_cfg.a_max,
            rms_norm=ssm_cfg.rms_norm,
            rms_eps=ssm_cfg.rms_eps,
            learnable_s0=ssm_cfg.learnable_s0,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        states: Optional[List[torch.Tensor]] = None,
        pos_offset: int = 0,
        return_states: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        x = self.embedding(input_ids, return_ports=False)
        x = x.float()

        if self.positional is not None:
            x = self.positional(x, offset=pos_offset)

        if states is None:
            states = [None] * len(self.blocks)
        if len(states) != len(self.blocks):
            raise ValueError("states length must match number of layers.")

        new_states = []
        for idx, block in enumerate(self.blocks):
            x, new_state = block(x, states[idx])
            new_states.append(new_state)

        logits = self.lm_head(x)
        if return_states:
            return logits, new_states
        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        eos_token_id: Optional[int] = None,
        tokenizer: Optional[object] = None,
        return_states: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        B, T = input_ids.shape
        if eos_token_id is None and tokenizer is not None:
            eos_token_id = getattr(tokenizer, "eos_id", None)
        logits, states = self.forward(
            input_ids, states=None, pos_offset=0, return_states=True
        )
        next_token = sample_logits(
            logits[:, -1, :], temperature=temperature, top_k=top_k, top_p=top_p
        )

        if max_new_tokens <= 0:
            if return_states:
                return input_ids, states
            return input_ids

        alive = torch.ones(B, dtype=torch.bool, device=input_ids.device)
        pos_offset = T
        generated = []

        for step in range(max_new_tokens):
            token_in = next_token
            if eos_token_id is not None and not alive.all():
                eos_fill = torch.full_like(token_in, eos_token_id)
                token_in = torch.where(alive, token_in, eos_fill)

            logits, new_states = self.forward(
                token_in.unsqueeze(1),
                states=states,
                pos_offset=pos_offset,
                return_states=True,
            )

            if eos_token_id is not None and not alive.all():
                mask = alive.unsqueeze(1).float()
                new_states = [
                    ns * mask + os * (1.0 - mask)
                    for ns, os in zip(new_states, states)
                ]

            generated.append(token_in)
            pos_offset += 1
            states = new_states

            if eos_token_id is not None:
                alive = alive & (token_in != eos_token_id)
                if not alive.any():
                    break

            next_token = sample_logits(
                logits[:, -1, :], temperature=temperature, top_k=top_k, top_p=top_p
            )

        if generated:
            gen_tokens = torch.stack(generated, dim=1)
            output_ids = torch.cat([input_ids, gen_tokens], dim=1)
        else:
            output_ids = input_ids

        if return_states:
            return output_ids, states
        return output_ids
