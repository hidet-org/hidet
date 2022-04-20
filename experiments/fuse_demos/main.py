from hidet.backend import build
from hidet.implement import implement
# from hidet.runtime.value import randn, empty, scalar, zeros, full, from_list
from hidet.tos.tensor import randn, empty, ones

from hidet.ir.dialects.compute import tensor_input, compute, reduce
from hidet.ir.task import Task, Grid
from hidet.ir.functors import inline_compute
from hidet.ir.primitives import cuda_max


def tuplize(v):
    if isinstance(v, (list, tuple)):
        return tuple(v)
    return v, v


def norm_pad(v):
    if isinstance(v, int):
        return [v, v, v, v]
    elif isinstance(v, (list, tuple)):
        if len(v) == 2:
            return [v[0], v[1], v[0], v[1]]
        elif len(v) == 4:
            return v
    raise NotImplementedError()


def conv2d_task(name, batch_size, in_channels, height, width, out_channels, kernel, padding, stride):
    kernel, padding, stride = tuplize(kernel), norm_pad(padding), tuplize(stride)
    input = tensor_input('input', 'float32', [batch_size, in_channels, height, width], scope='global')
    weight = tensor_input('weight', 'float32', [out_channels, in_channels, kernel[0], kernel[1]], scope='global')
    padded = compute(
        name='pad',
        shape=[batch_size, in_channels, height + padding[0] + padding[2], width + padding[1] + padding[3]],
        fcompute=lambda n, c, h, w: input.protect_read(indices=[n, c, h - padding[0], w - padding[1]], default_value=0.0)
    )
    out_height = (height + padding[0] + padding[2] - kernel[0]) // stride[0] + 1
    out_width = (width + padding[1] + padding[3] - kernel[1]) // stride[1] + 1
    bias = tensor_input('bias', 'float32', [batch_size, out_channels, out_height, out_width], scope='global')
    output = compute(
        name='out',
        shape=[batch_size, out_channels, out_height, out_width],
        fcompute=lambda n, c, h, w: cuda_max(
            reduce(
                shape=[in_channels, kernel[0], kernel[1]],
                fcompute=lambda rc, xx, yy: padded[n, rc, h * stride[0] + xx, w * stride[1] + yy] * weight.protect_read(indices=[c, rc, xx, yy], default_value=0.0),
                reduce_type='sum')
            +
            bias[0, c, 0, 0],
            0.0),
        scope='global'
    )
    output = inline_compute(output)
    return Task(
        name=name,
        computation=output,
        params=[input, weight, bias, output],
        worker=Grid()
    )


def fuse_conv2d_demo():
    name = 'conv_bias'
    n, rc, h, w = 1, 64, 32, 32
    kx, ky = 3, 3
    px, py = 1, 1
    sx, sy = 2, 2
    c = 64
    p = (h + 2 * px - kx) // sx + 1
    q = (w + 2 * py - ky) // sy + 1
    task = conv2d_task(name, batch_size=n, in_channels=rc, height=h, width=w, out_channels=c, kernel=(kx, ky), padding=(px, py), stride=(sx, sy))
    ir_module = implement(task)
    module = build(ir_module, output_dir=f'./outs/fuse/{name}', keep_ir=True)
    func = module[name]
    x = randn(shape=[n, rc, h, w])
    w = ones(shape=[c, rc, kx, ky])
    b = randn(shape=[1, c, 1, 1])
    y = empty(shape=[n, c, p, q])
    func(x, w, b, y)
    # print(x)
    # print(w)
    # print(b)
    # print(y)


if __name__ == '__main__':
    fuse_conv2d_demo()
