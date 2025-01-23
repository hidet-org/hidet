import sys
import argparse
import torch
from hidet.testing.torch_utils import bench_model, Backend


# MATMUL BENCHMARKS #
class torch_matmul(torch.nn.Module):
    def __init__(self):
        super(torch_matmul, self).__init__()

    def forward(self, a, b):
        return torch.matmul(a, b)


def create_model_matmul(params: str, dtype: torch.dtype):
    a_shape, b_shape = params.split(',')
    a_shape = [int(s) for s in a_shape.split('x')]
    b_shape = [int(s) for s in b_shape.split('x')]
    a = torch.randn(*a_shape, dtype=dtype, device='cuda')
    b = torch.randn(*b_shape, dtype=dtype, device='cuda')
    model = torch_matmul()
    return model, [a, b]


# CONV BENCHMARKS #
class torch_conv2d(torch.nn.Module):
    def __init__(self, w_shape, dtype: torch.dtype):
        super(torch_conv2d, self).__init__()
        self.w = torch.randn(*w_shape, dtype=dtype, device='cuda')

    def forward(self, x):
        return torch.nn.functional.conv2d(x, self.w)


def create_model_conv2d(params: str, dtype: torch.dtype):
    x_shape, w_shape = params.split(',')
    x_shape = [int(s) for s in x_shape.split('x')]
    w_shape = [int(s) for s in w_shape.split('x')]
    x = torch.randn(*x_shape, dtype=dtype, device='cuda')
    model = torch_conv2d(w_shape, dtype)
    return model, [x]


# ATTENTION BENCHMARKS #
class torch_attn(torch.nn.Module):
    def __init__(self, mask_shape=None):
        super(torch_attn, self).__init__()
        if mask_shape:
            self.mask = torch.randn(*mask_shape, dtype=torch.float16, device='cuda')
        else:
            self.mask = None

    def forward(self, q, k, v):
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=self.mask)


def create_model_attn(params: str, dtype: torch.dtype) -> float:
    bs, seqlen, nhead, hdim = [int(s) for s in params.split('x')]
    q_shape = [bs, nhead, seqlen, hdim]
    q = torch.randn(*q_shape, dtype=dtype, device='cuda')
    k = torch.randn(*q_shape, dtype=dtype, device='cuda')
    v = torch.randn(*q_shape, dtype=dtype, device='cuda')
    model = torch_attn()
    return model, [q, k, v]


def create_model_attn_mask_add(params: str, dtype: torch.dtype) -> float:
    bs, seqlen, nhead, hdim = [int(s) for s in params.split('x')]
    q_shape = [bs, nhead, seqlen, hdim]
    mask_shape = [bs, nhead, seqlen, seqlen]
    q = torch.randn(*q_shape, dtype=dtype, device='cuda')
    k = torch.randn(*q_shape, dtype=dtype, device='cuda')
    v = torch.randn(*q_shape, dtype=dtype, device='cuda')
    model = torch_attn(mask_shape=mask_shape)
    return model, [q, k, v]


# REDUCE #
class torch_sum(torch.nn.Module):
    def __init__(self, axis):
        super(torch_sum, self).__init__()
        self.axis = axis

    def forward(self, x):
        return torch.sum(x, dim=self.axis)


def create_model_reduce(params: str, dtype):
    x_shape, axis = params.split(',', maxsplit=1)
    start = axis.find('axis=[') + len('axis=[')
    end = axis.find(']', start)
    axis = [int(s) for s in axis[start:end].split(',')]
    x_shape = [int(s) for s in x_shape.split('x')]
    x = torch.randn(*x_shape, dtype=dtype, device='cuda')
    model = torch_sum(axis=axis)
    return model, [x]


# RESHAPE #
class torch_reshape(torch.nn.Module):
    def __init__(self, shape):
        super(torch_reshape, self).__init__()
        self.new_shape = shape

    def forward(self, x):
        return torch.reshape(x, self.new_shape)


def create_model_reshape(params: str, dtype):
    input_shape, output_shape = params.split(',', maxsplit=1)
    input_shape = [int(s) for s in input_shape.split('x')]
    output_shape = [int(s) for s in output_shape.split('x')]
    x = torch.randn(*input_shape, dtype=dtype, device='cuda')
    model = torch_reshape(output_shape)
    return model, [x]


# TRANSPOSE 2D #
class torch_transpose(torch.nn.Module):
    def __init__(self, input_shape, dim0, dim1):
        super(torch_transpose, self).__init__()
        self.input_shape = input_shape
        self.dim0 = int(dim0)
        self.dim1 = int(dim1)

    def forward(self, x):
        return torch.transpose(x, self.dim0, self.dim1).flatten()


def create_model_transpose(params: str, dtype):
    input_shape, dim0, dim1 = params.split(',', maxsplit=2)
    input_shape = [int(s) for s in input_shape.split('x')]
    x = torch.randn(*input_shape, dtype=dtype, device='cuda')
    model = torch_transpose(input_shape, dim0, dim1)
    return model, [x]


# LINEAR
class SimpleLinearModule(torch.nn.Module):
    def __init__(self, in_f, out_f, dtype):
        super(SimpleLinearModule, self).__init__()
        self.model = torch.nn.Linear(in_f, out_f, dtype=dtype, device='cuda')

    def forward(self, x):
        # x: (m, in_features)
        return self.model(x)  # (m, out_features)


def create_model_linear_dynamic(shape, dtype):
    return create_model_linear(shape, dtype, dynamic=True)


def create_model_linear_static(shape, dtype):
    return create_model_linear(shape, dtype, dynamic=False)


def create_model_linear(shape: str, dtype, dynamic=True):
    """
    Benchmark linear layers.
    Each shape is {m}x{in_features}x{out_features}.
    We'll simulate a linear layer: (m, in_features) * (in_features, out_features) -> (m, out_features)
    where m can be a dynamic dimension.
    """
    m, in_features, out_features = tuple(int(s) for s in shape.split('x'))

    model = SimpleLinearModule(in_features, out_features, dtype).cuda()
    example_inputs = torch.randn((m, in_features), device='cuda', dtype=dtype)
    if dynamic:
        torch._dynamo.mark_dynamic(example_inputs, 0)
    return model, [example_inputs]


# Main benchmark function for ops.
# Calls bench_model
def bench_op(operator, params, dtype, backend, mode):
    comp_backend = Backend(backend, mode, dtype)
    dtype = getattr(torch, dtype)

    model_creator = getattr(sys.modules[__name__], "create_model_" + operator)
    model, model_inputs = model_creator(params, dtype)
    model = model.eval().to(dtype).cuda()
    with torch.no_grad(), torch.autocast("cuda"):
        opt_model = comp_backend.compile(model)
        latency = bench_model(opt_model, model_inputs)

    return latency


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Benchmark Operators')
    parser.add_argument('operator', type=str, help='Specify operator. E.g., matmul_f16')
    parser.add_argument(
        '--params', type=str, help='Specify Input Parameters. Different operators have different formats.'
    )
    parser.add_argument('--dtype', type=str, default='float16', help='Specify precision. E.g., float32')
    parser.add_argument('--backend', type=str, default='hidet', help='torch.compile backend: hidet or max-autotune')
    parser.add_argument('--mode', type=str, default='max-autotune', help='Unused')

    args = parser.parse_args()

    operator, dtype, backend, mode = args.operator, args.dtype, args.backend, args.mode
    params = args.params
    latency = bench_op(operator, params, dtype, backend, mode)
    print(latency)
