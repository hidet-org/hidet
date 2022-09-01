import numpy as np
import tvm
import tvm.contrib
from tvm.relay.backend.executor_factory import ExecutorFactoryModule
from tvm.contrib import graph_executor
from tvm import relay, tir, te
from tvm.te import ComputeOp, PlaceholderOp, Schedule
import hidet
from hidet.utils.py import prod
import hidet.utils.tvm_utils


def conv2d_nchw_compute(batch, in_channel, in_height, in_width, stride_h, stride_w, padding_h, padding_w, out_channel, kernel_h, kernel_w):
    # compute the output shape
    out_height = (in_height - kernel_h + 2 * padding_h) // stride_h + 1
    out_width = (in_width - kernel_w + 2 * padding_w) // stride_w + 1
    # compute graph
    input = te.placeholder(shape=(batch, in_channel, in_height, in_width), dtype='float32', name='input')
    weight = te.placeholder(shape=(out_channel, in_channel, kernel_h, kernel_w), dtype='float32', name='weight')
    padded = te.compute(
        shape=(batch, in_channel, in_height + 2 * padding_h, in_width + 2 * padding_w),
        fcompute=lambda nn, ff, yy, xx: tir.if_then_else(cond=tir.all(yy - padding_h >= 0, yy - padding_h < in_height,
                                                                      xx - padding_w >= 0, xx - padding_w < in_width),
                                                         t=input[nn, ff, yy - padding_h, xx - padding_w],
                                                         f=tir.const(0.0)),
        name='pad'
    )
    rc = te.reduce_axis((0, in_channel), name="rc")
    ry = te.reduce_axis((0, kernel_h), name="ry")
    rx = te.reduce_axis((0, kernel_w), name="rx")
    conv = te.compute(
        shape=(batch, out_channel, out_height, out_width),
        fcompute=lambda nn, ff, yy, xx: te.sum(
            padded[nn, rc, yy * stride_h + ry, xx * stride_w + rx] * weight[ff, rc, ry, rx], axis=[rc, ry, rx]
        ),
        name='conv'
    )
    return input, weight, conv


def split(s, stage, axis, factors):
    split_sizes = [prod(factors[i:]) for i in range(len(factors))]
    sub_axes = []
    for split_size in split_sizes:
        sub_axis, axis = s[stage].split(axis, factor=split_size)
        sub_axes.append(sub_axis)
    sub_axes.append(axis)
    return sub_axes


def conv2d_nchw_direct_schedule(conv):
    # [('tile_f', [-1, 1, 16, 1]), ('tile_y', [-1, 1, 2, 1]), ('tile_x', [-1, 1, 1, 14]), ('tile_rc', [-1, 4, 1]), ('tile_ry', [-1, 1, 3]), ('tile_rx', [-1, 1, 3]), ('auto_unroll_max_step', 1500), ('unroll_explicit', 1)],None,100867170
    tile_f = [1, 16, 1]
    tile_y = [1, 2, 1]
    tile_x = [1, 1, 14]
    tile_rc = [1]
    tile_ry = [3]
    tile_rx = [3]

    s = te.create_schedule([conv.op])

    # n, f, y, x = s[conv].op.axis
    # rc, ry, rx = s[conv].op.reduce_axis
    # cfg.define_split("tile_f", f, num_outputs=4)
    # cfg.define_split("tile_y", y, num_outputs=4)
    # cfg.define_split("tile_x", x, num_outputs=4)
    # cfg.define_split("tile_rc", rc, num_outputs=2)
    # cfg.define_split("tile_ry", ry, num_outputs=2)
    # cfg.define_split("tile_rx", rx, num_outputs=2)

    pad_data, kernel = s[conv].op.input_tensors

    s[pad_data].compute_inline()
    if isinstance(kernel.op, tvm.te.ComputeOp) and "dilate" in kernel.op.tag:
        s[kernel].compute_inline()

    output = conv
    OL = s.cache_write(conv, "local")

    # create cache stage
    AA = s.cache_read(pad_data, "shared", [OL])
    WW = s.cache_read(kernel, "shared", [OL])

    # tile and bind spatial axes
    n, f, y, x = s[output].op.axis
    kernel_scope, n = s[output].split(n, nparts=1)

    bf, vf, tf, fi = split(s, output, f, tile_f)  # s[output].split(f)  # cfg["tile_f"].apply(s, output, f)
    by, vy, ty, yi = split(s, output, y, tile_y)  # s[output].split(y)  # cfg["tile_y"].apply(s, output, y)
    bx, vx, tx, xi = split(s, output, x, tile_x)  # s[output].split(x)  # cfg["tile_x"].apply(s, output, x)

    bf = s[output].fuse(n, bf)
    s[output].bind(bf, te.thread_axis("blockIdx.z"))
    s[output].bind(by, te.thread_axis("blockIdx.y"))
    s[output].bind(bx, te.thread_axis("blockIdx.x"))
    s[output].bind(vf, te.thread_axis("vthread"))
    s[output].bind(vy, te.thread_axis("vthread"))
    s[output].bind(vx, te.thread_axis("vthread"))
    s[output].bind(tf, te.thread_axis("threadIdx.z"))
    s[output].bind(ty, te.thread_axis("threadIdx.y"))
    s[output].bind(tx, te.thread_axis("threadIdx.x"))
    s[output].reorder(bf, by, bx, vf, vy, vx, tf, ty, tx, fi, yi, xi)
    s[OL].compute_at(s[output], tx)

    # tile reduction axes
    n, f, y, x = s[OL].op.axis
    rc, ry, rx = s[OL].op.reduce_axis
    rco, rci = split(s, OL, rc, tile_rc)  # cfg["tile_rc"].apply(s, OL, rc)
    ryo, ryi = split(s, OL, ry, tile_ry)  # cfg["tile_ry"].apply(s, OL, ry)
    rxo, rxi = split(s, OL, rx, tile_rx)  # cfg["tile_rx"].apply(s, OL, rx)
    s[OL].reorder(rco, ryo, rxo, rci, ryi, rxi, n, f, y, x)
    s[AA].compute_at(s[OL], rxo)
    s[WW].compute_at(s[OL], rxo)

    # cooperative fetching
    for load in [AA, WW]:
        n, f, y, x = s[load].op.axis
        fused = s[load].fuse(n, f, y, x)
        tz, fused = s[load].split(fused, nparts=tile_f[1])  # s[load].split(fused, nparts=cfg["tile_f"].size[2])
        ty, fused = s[load].split(fused, nparts=tile_y[1])  # s[load].split(fused, nparts=cfg["tile_y"].size[2])
        tx, fused = s[load].split(fused, nparts=tile_x[1])  # s[load].split(fused, nparts=cfg["tile_x"].size[2])
        s[load].bind(tz, te.thread_axis("threadIdx.z"))
        s[load].bind(ty, te.thread_axis("threadIdx.y"))
        s[load].bind(tx, te.thread_axis("threadIdx.x"))
    return s


def demo_conv2d_te():
    settings = [
        {'batch': 1, 'in_channel': 256, 'in_height': 14, 'in_width': 14, 'stride_h': 1, 'stride_w': 1, 'padding_h': 1, 'padding_w': 1, 'out_channel': 256, 'kernel_h': 3, 'kernel_w': 3}
        # {'batch': 1, 'in_channel': 128, 'in_height': 128, 'in_width': 128, 'stride_h': 1, 'stride_w': 1, 'padding_h': 1, 'padding_w': 1, 'out_channel': 128, 'kernel_h': 3, 'kernel_w': 3}
        # {'batch': 1, 'in_channel': 128, 'in_height': 128, 'in_width': 128, 'stride_h': 2, 'stride_w': 2, 'padding_h': 1, 'padding_w': 1, 'out_channel': 128, 'kernel_h': 3, 'kernel_w': 3}
        # {'batch': 1, 'in_channel': 64, 'in_height': 64, 'in_width': 64, 'stride_h': 2, 'stride_w': 2, 'padding_h': 1, 'padding_w': 1, 'out_channel': 64, 'kernel_h': 3, 'kernel_w': 3}
    ]
    input, weight, conv = conv2d_nchw_compute(**settings[0])
    sch = conv2d_nchw_direct_schedule(conv)
    with open('./outs/tvm_lowered.txt', 'w') as f:
        lowered = str(tvm.lower(sch, [input, weight, conv], name='conv2d', simple_mode=True))
        f.write(lowered)
    # module = tvm.build(sch, [input, weight, conv], target='cuda', name='conv2d')


def get_conv_model(batch_size, in_channels, h, w, out_channels, kernel, padding, stride):
    x = relay.var('x', shape=(batch_size, in_channels, h, w))
    w = relay.var('w', shape=(out_channels, in_channels, *kernel))
    y = relay.nn.conv2d(x, w, stride, padding)
    func = relay.Function(params=[x, w], body=y)
    return tvm.ir.IRModule.from_expr(func)


def demo_conv2d():
    ir_module = get_conv_model(1, 3, 224, 224, 64, (7, 7), (3, 3), stride=(2, 2))
    hidet.utils.tvm_utils.dump_relay_cuda_code(ir_module, out_dir='./outs')
    graph_module: ExecutorFactoryModule = relay.build(ir_module, target='cuda')
    x = tvm.nd.empty(shape=(1, 3, 224, 224), dtype='float32', device=tvm.cuda())
    w = tvm.nd.empty(shape=(64, 3, 7, 7), dtype='float32', device=tvm.cuda())
    executor = graph_executor.GraphModule(graph_module['default'](tvm.cuda()))
    y = executor.run(x=x, w=w)


def demo_vthread():
    a = te.placeholder((100,), name='a')
    b = te.compute((100,), fcompute=lambda i: a[i], name='b')
    s = te.create_schedule([b.op])
    io, ii = s[b].split(b.op.axis[0], factor=10)
    s[b].bind(io, te.thread_axis('vthread'))
    print(tvm.lower(s, [a, b], simple_mode=True))


def demo_softmax():
    batch_size = 20
    features = 2000
    x = relay.var('x', shape=(batch_size, features))
    y = relay.nn.softmax(x, axis=1)
    # y = relay.nn.fast_softmax(x, axis=1)
    func = relay.Function(params=[x], body=y)
    ir_module = tvm.ir.IRModule.from_expr(func)
    hidet.utils.tvm_utils.dump_relay_cuda_code(ir_module, out_dir='./outs/softmax')


def demo_pool2d():
    n, c, h, w = 1, 64, 112, 112
    x = relay.var('x', shape=(n, c, h, w))
    y = relay.nn.max_pool2d(x, pool_size=(3, 3), strides=(2, 2), padding=(1, 1))
    func = relay.Function(params=[x], body=y)
    ir_module = tvm.ir.IRModule.from_expr(func)
    hidet.utils.tvm_utils.dump_relay_cuda_code(ir_module, out_dir='./outs/max_pool2d')


def demo_batch_norm():
    n, c, h, w = 1, 64, 112, 112
    x = relay.var('x', shape=[n, c, h, w])
    running_mean = relay.var('running_mean', shape=[c])
    running_var = relay.var('running_var', shape=[c])
    beta = relay.var('beta', shape=[c])
    gamma = relay.var('gamma', shape=[c])
    y, y1, y2 = relay.nn.batch_norm(x, gamma=gamma, beta=beta, moving_mean=running_mean, moving_var=running_var, axis=1)
    s = y + relay.reshape(y1, [1, c, 1, 1]) + relay.reshape(y2, [1, c, 1, 1])
    func = relay.Function(params=[x, running_mean, running_var, beta, gamma], body=s)
    ir_module = tvm.ir.IRModule.from_expr(func)
    hidet.utils.tvm_utils.dump_relay_cuda_code(ir_module, out_dir='./outs/bn')


def demo_rsqrt():
    batch_size = 20
    features = 2000
    x = relay.var('x', shape=(batch_size, features))
    y = relay.rsqrt(x)
    func = relay.Function(params=[x], body=y)
    ir_module = tvm.ir.IRModule.from_expr(func)
    hidet.utils.tvm_utils.dump_relay_cuda_code(ir_module, out_dir='./outs/rsqrt')


def demo_gather():
    data = relay.var('data', shape=(1234, 567))
    indices = relay.var('indices', shape=(512,), dtype='int64')
    gathered = relay.take(data, indices, axis=0)
    func = relay.Function(params=[data, indices], body=gathered)
    ir_module = tvm.ir.IRModule.from_expr(func)
    hidet.utils.tvm_utils.dump_relay_cuda_code(ir_module, out_dir='./outs/take')


def demo_variance():
    data = relay.var('data', shape=(1234, 567))
    variance = relay.variance(data, axis=1, keepdims=False)
    func = relay.Function(params=[data], body=variance)
    ir_module = tvm.ir.IRModule.from_expr(func)
    hidet.utils.tvm_utils.dump_relay_cuda_code(ir_module, out_dir='./outs/variance')


def demo_reduce_mean():
    # data = relay.var('data', shape=(1, 55, 131072))
    data = relay.var('data', shape=(1, 131072, 55))
    # reduced = relay.mean(data, axis=2, keepdims=True)
    reduced = relay.mean(data, axis=1, keepdims=True)
    func = relay.Function(params=[data], body=reduced)
    ir_module = tvm.ir.IRModule.from_expr(func)
    hidet.utils.tvm_utils.dump_relay_cuda_code(ir_module, out_dir='./outs/reduce_mean')


def demo_winograd_conv2d():
    x = relay.var('x', shape=(1, 128, 28, 28), dtype='float16')
    # w = relay.var('w', shape=(128, 128, 3, 3))
    w = relay.const(np.random.randn(128, 128, 3, 3).astype(np.float), dtype='float16')
    y = relay.nn.conv2d(x, w, strides=1, padding=1, kernel_size=3, out_dtype='float32')
    func = relay.Function(params=[x], body=y)
    ir_module = tvm.ir.IRModule.from_expr(func)
    print(ir_module)
    hidet.utils.tvm_utils.dump_relay_cuda_code(
        ir_module=ir_module,
        out_dir='./outs/winograd'
    )


def demo_conv2d_hwnc():
    n, c, h, w = 1, 128, 28, 28
    x = relay.var('x', shape=(n, c, h, w), dtype='float16')
    # w = relay.var('w', shape=(128, 128, 3, 3))
    w = relay.const(np.random.randn(c, c, 3, 3).astype(np.float), dtype='float16')
    y = relay.nn.conv2d(x, w, strides=1, padding=1, kernel_size=3, out_dtype='float32',
                        )
    func = relay.Function(params=[x], body=y)
    ir_module = tvm.ir.IRModule.from_expr(func)
    print(ir_module)
    hidet.utils.tvm_utils.dump_relay_cuda_code(
        ir_module=ir_module,
        out_dir='./outs/hwnc'
    )


def demo_double_buffer():
    import tvm
    from tvm import te

    A = te.placeholder([10], name='A')
    B = te.placeholder([10], name='B')
    C = te.compute([10], lambda i: A[i] + B[i], 'C')

    s: te.Schedule = te.create_schedule(C.op)
    AS = s.cache_read(A, 'shared', [C])
    BS = s.cache_read(B, 'shared', [C])
    s[AS].double_buffer()
    s[BS].double_buffer()

    print(tvm.lower(s, [A, B, C], simple_mode=True))
    # fadd = tvm.build(s, [A, B, C], target='cuda', name='myadd')


def demo_dense():
    m, k, n = 2024, 2024, 2024
    x = relay.var('x', shape=(m, k), dtype='float32')
    y = relay.var('y', shape=(k, n), dtype='float32')
    z = relay.nn.matmul(x, y)
    func = relay.Function(params=[x, y], body=z)
    ir_module = tvm.ir.IRModule.from_expr(func)
    print(ir_module)
    hidet.utils.tvm_utils.dump_relay_cuda_code(
        ir_module=ir_module,
        out_dir='./outs/dense_{}_{}_{}'.format(m, n, k),
        opt_level=0
    )


def demo_dense_large():
    from tvm import topi
    fcompute, fschedule = topi.gpu.dense_large_batch, topi.gpu.schedule_dense_large_batch

    batch_size, in_dim, out_dim = 1024, 1024, 1024

    A = te.placeholder((batch_size, in_dim), name="A", dtype='float32')
    B = te.placeholder((out_dim, in_dim), name="B", dtype='float32')
    C = te.placeholder((out_dim,), name="C", dtype='float32')

    with tvm.target.Target('cuda'):
        D = fcompute(A, B)
        s = fschedule([D])

    func = tvm.driver.build(s, [A, B, C], target='cuda')
    with open('./outs/large_dense.cu', 'w') as f:
        f.write(func.imported_modules[0].get_source())
    # hidet.utils.tvm_utils.dump_relay_cuda_code(
    #     ir_module=ir_module,
    #     out_dir='./outs/large_dense',
    # )
    # a = tvm.nd.array(a_np, dev)
    # b = tvm.nd.array(b_np, dev)
    # c = tvm.nd.array(c_np, dev)
    # d = tvm.nd.array(np.zeros(get_const_tuple(D.shape), dtype=out_dtype), dev)
    # f = tvm.build(s, [A, B, C, D], target, name="dense")
    # f(a, b, c, d)
    # tvm.testing.assert_allclose(d.numpy(), d_np, **tol)


def demo_double_buffer_v2():
    import tvm
    from tvm import topi
    from tvm import te

    A = te.placeholder((100,), name='A')
    B = topi.nn.relu(A)

    s: te.Schedule = te.create_schedule(B.op)
    AA = s.cache_read(A, 'shared', [B])
    s[AA].compute_at(s[B], B.op.axis[0])
    s[AA].double_buffer()
    s[B].bind(B.op.axis[0], te.thread_axis('blockIdx.x'))
    func = tvm.build(s, [A, B], target='cuda', name='func')
    print(tvm.lower(s, [A, B], simple_mode=True))
    print(func.imported_modules[0].get_source())


def demo_depthwise():
    # n, c, h, w, kx, ky, sx, sy = out_shape 1 32 112 112 kernel 3 3 stride 1 1
    # n, c, h, w, kx, ky, sx, sy = out_shape 1 96 56 56 kernel 3 3 stride 2 2
    # n, c, h, w, kx, ky, sx, sy = out_shape 1 144 56 56 kernel 3 3 stride 1 1
    # n, c, h, w, kx, ky, sx, sy = out_shape 1 144 28 28 kernel 3 3 stride 2 2
    # n, c, h, w, kx, ky, sx, sy = out_shape 1 192 28 28 kernel 3 3 stride 1 1
    # n, c, h, w, kx, ky, sx, sy = out_shape 1 192 28 28 kernel 3 3 stride 1 1
    # n, c, h, w, kx, ky, sx, sy = out_shape 1 192 14 14 kernel 3 3 stride 2 2
    # n, c, h, w, kx, ky, sx, sy = out_shape 1 384 14 14 kernel 3 3 stride 1 1
    # n, c, h, w, kx, ky, sx, sy = out_shape 1 384 14 14 kernel 3 3 stride 1 1
    # n, c, h, w, kx, ky, sx, sy = out_shape 1 384 14 14 kernel 3 3 stride 1 1
    # n, c, h, w, kx, ky, sx, sy = out_shape 1 384 14 14 kernel 3 3 stride 1 1
    # n, c, h, w, kx, ky, sx, sy = out_shape 1 576 14 14 kernel 3 3 stride 1 1
    # n, c, h, w, kx, ky, sx, sy = out_shape 1 576 14 14 kernel 3 3 stride 1 1
    # n, c, h, w, kx, ky, sx, sy = out_shape 1 576 7 7 kernel 3 3 stride 2 2
    # n, c, h, w, kx, ky, sx, sy = out_shape 1 960 7 7 kernel 3 3 stride 1 1
    # n, c, h, w, kx, ky, sx, sy = out_shape 1 960 7 7 kernel 3 3 stride 1 1
    # n, c, h, w, kx, ky, sx, sy = out_shape 1 960 7 7 kernel 3 3 stride 1 1

    # n, c, h, w, r, s = 1, 960, 7, 7, 3, 3
    # n, c, h, w, r, s = 1, 576, 14, 14, 3, 3
    # n, c, h, w, r, s = 1, 192, 28, 28, 3, 3
    n, c, h, w, r, s = 1, 384, 14, 14, 3, 3
    x = relay.var('x', shape=(n, c, h, w), dtype='float32')
    y = relay.var('y', shape=(c, 1, r, s), dtype='float32')
    z = relay.nn.conv2d(x, y, strides=(1, 1), padding=(1, 1), groups=c)
    func = relay.Function(params=[x, y], body=z)
    ir_module = tvm.ir.IRModule.from_expr(func)
    print(ir_module)
    hidet.utils.tvm_utils.dump_relay_cuda_code(
        ir_module=ir_module,
        out_dir='./outs/depthwise_{}_{}_{}_{}_{}_{}'.format(n, c, h, w, r, s)
    )


if __name__ == '__main__':
    # demo_vthread()
    # demo_conv2d_te()
    # demo_softmax()
    # demo_pool2d()
    # demo_batch_norm()
    # demo_rsqrt()
    # demo_gather()
    # demo_variance()
    # demo_reduce_mean()
    # demo_winograd_conv2d()
    # demo_double_buffer_v2()
    # demo_dense()
    # demo_dense_large()
    demo_depthwise()
