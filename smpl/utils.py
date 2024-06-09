import torch
from functools import partial

def normalize_tensor(x: torch.Tensor, dim=-1, return_norm=False):
    r"""
    Normalize a tensor in a specific dimension to unit norm. (torch)

    :param x: Tensor in any shape.
    :param dim: The dimension to be normalized.
    :param return_norm: If True, norm(length) tensor will also be returned.
    :return: Tensor in the same shape. If return_norm is True, norm tensor in shape [*, 1, *] (1 at dim)
             will also be returned (keepdim=True).
    """
    norm = x.norm(dim=dim, keepdim=True)
    normalized_x = x / norm
    return normalized_x if not return_norm else (normalized_x, norm)


def append_value(x: torch.Tensor, value: float, dim=-1):
    r"""
    Append a value to a tensor in a specific dimension. (torch)

    e.g. append_value(torch.zeros(3, 3, 3), 1, dim=1) will result in a tensor of shape [3, 4, 3] where the extra
         part of the original tensor are all 1.

    :param x: Tensor in any shape.
    :param value: The value to be appended to the tensor.
    :param dim: The dimension to be expanded.
    :return: Tensor in the same shape except for the expanded dimension which is 1 larger.
    """
    app = torch.ones_like(x.index_select(dim, torch.tensor([0], device=x.device))) * value
    x = torch.cat((x, app), dim=dim)
    return x


append_zero = partial(append_value, value=0)
append_one = partial(append_value, value=1)