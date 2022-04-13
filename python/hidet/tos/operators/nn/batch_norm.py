from ..common import Tensor


def batch_norm_infer(x: Tensor, running_mean: Tensor, running_var: Tensor, epsilon=1e-5, axis=1) -> Tensor:
    assert len(x.shape) == 4 and axis == 1
    assert len(running_mean.shape) == 1 and len(running_var.shape) == 1
    assert x.shape[1] == running_mean.shape[0] == running_var.shape[0]
    running_mean = running_mean.unsqueeze([0, 2, 3])  # [1, c, 1, 1]
    running_var = running_var.unsqueeze([0, 2, 3])
    return (x - running_mean) * (running_var + epsilon).rsqrt()
