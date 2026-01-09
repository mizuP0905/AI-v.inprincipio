import torch

from cepta.config import CeptaConfig, GraphConfig, PositionalConfig, SSMConfig
from cepta.cepta_perceptron import CeptaPerceptronDense
from cepta.model import CeptaLM
from cepta.sampling import sample_logits
from cepta.synapse_graph import SynapseGraph


def build_small_config() -> CeptaConfig:
    cfg = CeptaConfig(
        vocab_size=32,
        P=8,
        alpha=2,
        K=4,
        num_layers=2,
        seed_base=7,
        use_ste=False,
        dale_mode=False,
    )
    cfg.graph_ssm = GraphConfig(P=cfg.P, alpha=cfg.alpha, K=cfg.K)
    cfg.graph_mlp = GraphConfig(P=cfg.P, alpha=cfg.alpha, K=cfg.K)
    cfg.ssm = SSMConfig(P=cfg.P, P_r=4, mode="mode1")
    cfg.pos = PositionalConfig(mode="A", max_seq_len=32)
    return cfg


def test_forward_backward_cpu():
    cfg = build_small_config()
    model = CeptaLM(cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (2, 5))
    logits, _ = model.forward(input_ids, return_states=True)
    loss = logits.sum()
    loss.backward()
    assert model.embedding.embedding.weight.grad is not None


def test_forward_backward_gpu():
    if not torch.cuda.is_available():
        return
    cfg = build_small_config()
    model = CeptaLM(cfg).cuda()
    input_ids = torch.randint(0, cfg.vocab_size, (2, 5)).cuda()
    logits, _ = model.forward(input_ids, return_states=True)
    loss = logits.sum()
    loss.backward()
    assert model.embedding.embedding.weight.grad is not None


def test_dale_mode_neuron_types_provided_and_auto():
    P, K, alpha = 4, 3, 2
    neuron_types = torch.tensor([1, 0, 1, 0])
    perceptron = CeptaPerceptronDense(
        P=P,
        K=K,
        alpha=alpha,
        dale_mode=True,
        neuron_types=neuron_types,
    )
    f = perceptron.compute_f()
    assert (f[neuron_types == 1] > 0).all()
    assert (f[neuron_types == 0] < 0).all()

    perceptron_auto = CeptaPerceptronDense(
        P=P,
        K=K,
        alpha=alpha,
        dale_mode=True,
        neuron_type_seed=123,
    )
    assert perceptron_auto.neuron_types.shape == (P, 1)
    x = torch.randn(2, 3, P, K)
    perceptron_auto(x)


def test_synapsegraph_unique_fallback():
    P, alpha, K = 2, 2, 8
    positions = torch.rand(P, 2)
    src_idx = SynapseGraph.generate_src_idx(
        positions_src=positions,
        positions_tgt=positions,
        alpha_source=alpha,
        K=K,
        lambda_local=1.0,
        rho_lr=0.2,
        r_hub=0.5,
        rho_hub=0.5,
        unique_per_target=True,
        allow_self=True,
        seed=42,
        same_layer=True,
    )
    assert src_idx.shape == (P, K)
    assert src_idx.max().item() < P * alpha


def test_generate_warmup_no_double_count():
    torch.manual_seed(0)
    cfg = build_small_config()
    model = CeptaLM(cfg)
    prompt = torch.randint(0, cfg.vocab_size, (1, 4))

    logits, states_prompt = model.forward(prompt, return_states=True)
    first_token = sample_logits(logits[:, -1, :], temperature=0.0)

    _, states_manual = model.forward(
        first_token.unsqueeze(1),
        states=states_prompt,
        pos_offset=prompt.shape[1],
        return_states=True,
    )

    _, states_gen = model.generate(
        prompt, max_new_tokens=1, temperature=0.0, return_states=True
    )

    for s_manual, s_gen in zip(states_manual, states_gen):
        assert torch.allclose(s_manual, s_gen)
