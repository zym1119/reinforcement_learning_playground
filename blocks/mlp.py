from typing import List, Optional

import torch.nn as nn


def build_mlp(input_dim: int, output_dim: int, hidden_dims: List[int],
              activation: str = 'relu', output_activation: Optional[str] = None) -> nn.Sequential:
    """
    构建通用 MLP 网络。

    Args:
        input_dim: 输入维度
        output_dim: 输出维度
        hidden_dims: 隐藏层维度列表，如 [64, 64]
        activation: 隐藏层激活函数 ('relu', 'tanh', 'silu')
        output_activation: 输出层激活函数（None 表示无激活）

    Returns:
        nn.Sequential 模型
    """
    activation_fn = _get_activation(activation)

    layers = []
    prev_dim = input_dim
    for h_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, h_dim))
        layers.append(activation_fn())
        prev_dim = h_dim
    layers.append(nn.Linear(prev_dim, output_dim))

    if output_activation:
        layers.append(_get_activation(output_activation)())

    model = nn.Sequential(*layers)
    _init_weights(model)
    return model


def _get_activation(name: str):
    """根据名称获取激活函数类"""
    activations = {
        'relu': nn.ReLU,
        'tanh': nn.Tanh,
        'silu': nn.SiLU,
        'gelu': nn.GELU,
        'softmax': lambda: nn.Softmax(dim=-1),
    }
    if name not in activations:
        raise ValueError(f"Unknown activation: {name}. Available: {list(activations.keys())}")
    return activations[name]


def _init_weights(model: nn.Module):
    """正交初始化"""
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=1.0)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
