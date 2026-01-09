import torch


def flatten_ports(y: torch.Tensor) -> torch.Tensor:
    # y: (B, T, P, alpha) -> (B, T, P*alpha)
    B, T, P, alpha = y.shape
    return y.reshape(B, T, P * alpha)


def unflatten_ports(x: torch.Tensor, P: int, alpha: int) -> torch.Tensor:
    # x: (B, T, P*alpha) -> (B, T, P, alpha)
    B, T, _ = x.shape
    return x.reshape(B, T, P, alpha)
