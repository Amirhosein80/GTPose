import torch
from functools import partial

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