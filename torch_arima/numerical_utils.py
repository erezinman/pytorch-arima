import torch


def causal_transposed_conv1d(x: torch.Tensor, weights: torch.Tensor):
    max_pos = -weights.shape[2] + 1 if weights.shape[2] > 1 else None
    return torch.conv_transpose1d(x, weights)[:, :, :max_pos]


def polymul(poly1: torch.Tensor, poly2: torch.Tensor) -> torch.Tensor:
    """

    Check that it's the same as numpy's:
    >>> import numpy as np
    >>> for i in range(100000):
    ...     lx, ly = np.random.choice(10, 2) + 1
    ...     x = np.random.uniform(-10, 10, lx)
    ...     y = np.random.uniform(-10, 10, ly)
    ...     res1 = np.polymul(x, y)
    ...     res2 = polymul(torch.tensor(x), torch.tensor(y)).flatten().numpy()
    ...
    ...     assert np.allclose(res1, res2, 1e-10, 1e-10)

    """
    return torch.conv_transpose1d(poly1.reshape(1, 1, -1), poly2.reshape(1, 1, -1))


def polypow(poly: torch.Tensor, power: int) -> torch.Tensor:
    result = torch.as_tensor([1], dtype=poly.dtype, device=poly.device)
    for _ in range(power):
        result = polymul(poly, result)

    return result
