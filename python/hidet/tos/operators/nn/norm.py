from ..common import Tensor
from ..basic.arithmatic import square
from ..basic.reduce import reduce_mean, reduce_sum


def batch_norm_infer(x: Tensor, running_mean: Tensor, running_var: Tensor, epsilon=1e-5, axis=1) -> Tensor:
    assert len(x.shape) == 4 and axis == 1
    assert len(running_mean.shape) == 1 and len(running_var.shape) == 1
    assert x.shape[1] == running_mean.shape[0] == running_var.shape[0]
    running_mean = running_mean.unsqueeze([0, 2, 3])  # [1, c, 1, 1]
    running_var = running_var.unsqueeze([0, 2, 3])
    return (x - running_mean) * (running_var + epsilon).rsqrt()


def instance_norm(x: Tensor, epsilon: float) -> Tensor:
    # todo: make it more efficient
    assert len(x.shape) >= 3
    dims = list(range(2, len(x.shape)))
    mean = reduce_mean(x, dims=dims, keep_dim=True)
    variance = reduce_sum(square(x), dims=dims, keep_dim=True) - square(reduce_sum(x, dims=dims, keep_dim=True))
    return (x - mean) / (variance + epsilon)

