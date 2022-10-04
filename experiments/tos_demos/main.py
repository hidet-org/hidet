from typing import List
import time
import numpy as np
import math

import hidet
import hidet.testing
from hidet import FlowGraph
from hidet.runtime import CudaGraph, CudaStream
from hidet.graph import Tensor, ops
from hidet.graph.tensor import symbol_like, randn, array
from hidet.utils import error_tolerance
from hidet.utils import nvtx_annotate


def demo_block(name: str):
    name2block = {
        'inception_head': hidet.testing.tos_models.inception.inception_head(),
        'inception_a': hidet.testing.tos_models.inception.inception_a(),
        'inception_b': hidet.testing.tos_models.inception.inception_b(),
        'inception_c': hidet.testing.tos_models.inception.inception_c(),
        'inception_d': hidet.testing.tos_models.inception.inception_d(),
        'inception_e': hidet.testing.tos_models.inception.inception_e(),
        'inception_tail': hidet.testing.tos_models.inception.inception_tail(),
        'inception_v3': hidet.testing.tos_models.inception.inception_v3(),
        'bert': hidet.testing.tos_models.bert.bert(),
    }
    model, inputs = name2block[name]
    symbol_inputs = [symbol_like(x) for x in inputs]
    symbol_outputs = model(*symbol_inputs)

    graph = hidet.trace_from(symbol_outputs, symbol_inputs)

    outputs = graph(*inputs)
    if not isinstance(outputs, (list, tuple)):
        outputs = [outputs]

    with hidet.tos.PassContext() as ctx:
        ctx.save_graph_instrument(f'./outs/{name}')
        graph_opt = hidet.tos.optimize(graph)

    outputs_opt = graph_opt(*inputs)
    if not isinstance(outputs_opt, (list, tuple)):
        outputs_opt = [outputs_opt]

    for y, y_opt in zip(outputs, outputs_opt):
        np.testing.assert_allclose(actual=y_opt.numpy(), desired=y.numpy(), atol=1e-2, rtol=1e-2)


def demo_basic_conv2d():
    model, inputs = hidet.testing.tos_models.inception.basic_conv2d(
        batch_size=1, in_channels=128, height=17, width=17, out_channels=128,
        kernel_size=[7, 1], padding=[3, 0], stride=1, groups=1
    )
    symbol_inputs = [symbol_like(x) for x in inputs]
    symbol_outputs = model(*symbol_inputs)

    graph = hidet.trace_from(symbol_outputs, symbol_inputs)

    y = model(*inputs)

    with hidet.tos.PassContext() as ctx:
        ctx.save_graph_instrument('./outs/basic_conv2d')
        graph_opt = hidet.tos.optimize(graph)

    y_opt = graph_opt(*inputs)
    np.testing.assert_allclose(actual=y_opt.numpy(), desired=y.numpy(), atol=1e-4, rtol=1e-4)


def demo_model_inputs(model, inputs):
    symbol_inputs = [symbol_like(x) for x in inputs]
    symbol_outputs = model(*symbol_inputs)

    graph = hidet.trace_from(symbol_outputs, symbol_inputs)

    y = model(*inputs)

    with hidet.tos.PassContext() as ctx:
        ctx.save_graph_instrument('./outs/custom')
        graph_opt = hidet.tos.optimize(graph)

    y_opt = graph_opt(*inputs)
    np.testing.assert_allclose(actual=y_opt.numpy(), desired=y.numpy(), atol=1e-4, rtol=1e-4)


def parallel_k_batched_matmul(a: Tensor, b: Tensor, mma: str = 'default'):
    from hidet.tos.ops.definitions.matmul.parallel_k_matmul import use_parallel_k, parallel_k_nparts, matmul
    import warnings
    k_size = a.shape[-1]
    batch_size, m_size, n_size = a.shape[0], a.shape[1], b.shape[2]
    if use_parallel_k(batch_size, m_size, n_size, k_size):
        nparts = parallel_k_nparts(batch_size, m_size, n_size, k_size)
        a = a.reshape([batch_size, m_size, nparts, k_size // nparts]).rearrange([[0, 2], [1], [3]])  # [batch_size * nparts, m_size, k_size // nparts]
        b = b.reshape([batch_size, nparts, k_size // nparts, n_size]).rearrange([[0, 1], [2], [3]])  # [batch_size * nparts, k_size // nparts, n_size]
        c1 = matmul(a, b, algo='direct', mma=mma)
        c = c1.reshape([batch_size, nparts, m_size, n_size]).sum(1)
        return c1, c
    else:
        warnings.warn('Please use use_parallel_k to check whether we should use parallel_k matmul first. Falling back to direct algorithm.')
        return matmul(a, b, algo='direct', mma=mma)


def demo_parallel_k_matmul():
    a = hidet.randn([1, 128, 768], dtype='float16')
    b = hidet.randn([1, 768, 2304], dtype='float16')
    c1 = ops.matmul(a.cast('float32'), b.cast('float32'), algo='direct', mma='simt')
    c2 = ops.matmul(a, b, algo='direct', mma='wmma_f16_f32')
    c3_1, c3 = parallel_k_batched_matmul(a, b, mma='wmma_f16_f32')
    np.testing.assert_allclose(actual=c2.numpy(), desired=c1.numpy(), rtol=1e-1, atol=1e-1)
    np.testing.assert_allclose(actual=c3.numpy(), desired=c1.numpy(), rtol=1e-1, atol=1e-1)


def demo_batched_matmul():
    # batch_size, m_size, n_size, k_size = 4, 128, 2304, 192
    batch_size, m_size, n_size, k_size = 4, 128, 2304, 192
    a = hidet.randn([batch_size, m_size, k_size]).cast('float16')
    b = hidet.randn([batch_size, k_size, n_size]).cast('float16')
    c1 = ops.matmul(a.cast('float32'), b.cast('float32'), algo='direct', mma='simt')
    c2 = ops.matmul(a, b, algo='direct', mma='wmma_f16_f16')

    print(np.count_nonzero(np.isnan(c1.numpy())))
    print(np.count_nonzero(np.isnan(c2.numpy())))

    np.testing.assert_allclose(actual=c2.numpy(), desired=c1.numpy(), rtol=1e-1, atol=1e-1)


def error_rank(actual: np.ndarray, desire: np.ndarray) -> int:
    for rank in range(-10, 10):
        e = 10 ** rank
        if np.allclose(actual, desire, atol=e, rtol=e):
            return rank
    return 10


def demo_batch_matmul_precision():
    batch_size, m_size, n_size, k_size = 1, 128, 2304, 768
    # tf32-f32
    # f16-f32
    # f16-f16
    # bf16-f32
    a = hidet.randn([batch_size, m_size, k_size])
    b = hidet.randn([batch_size, k_size, n_size])
    for dtype in [
        'float16',
        'bfloat16',
        'float32'
    ]:
        c = hidet.ops.matmul(a.cast(dtype).cast('float32'), b.cast(dtype).cast('float32'), algo='direct', mma='simt')
        c_np = c.numpy()
        for mma in [
            'simt',
            'wmma_f16_f16',
            'wmma_f16_f32',
            'wmma_bf16_f32',
            'wmma_tf32_f32'
        ]:
            aa = a.cast(dtype)
            bb = b.cast(dtype)
            cc = hidet.ops.matmul(aa, bb, algo='direct', mma=mma)
            cc_np = cc.numpy()
            # print('{:4}({:0.3f})'.format(error_rank(actual=cc_np, desire=c_np), (c_np - cc_np).max()), end=' ')
            # print('{:0.5f}'.format((c_np - cc_np).max()), end=' ')
            print('{:0.8f}'.format(abs(c_np - cc_np).astype(np.float32).mean()), end=' ')
        print()
    for v in ['float16', 'bfloat16', 'float32']:
        print(v)
    for v in ['simt', 'wmma_f16_f16', 'wmma_f16_f32', 'wmma_bf16_f32', 'wmma_tf32_f32']:
        print(v)


def demo_no_barrier():
    x = randn([3, 4])
    x_symbol = symbol_like(x)
    y1 = x_symbol + x_symbol
    y2 = y1 + x_symbol
    graph = hidet.trace_from(y2, x_symbol)
    with hidet.tos.PassContext() as ctx:
        ctx.save_graph_instrument('./outs/no_barrier')
        opt_graph = hidet.tos.optimize(graph)


def demo_barrier():
    x = randn([3, 4])
    x_symbol = symbol_like(x)
    y1 = x_symbol + x_symbol
    y1 = hidet.ops.barrier(y1)
    y2 = y1 + x_symbol
    graph = hidet.trace_from(y2, x_symbol)
    with hidet.tos.PassContext() as ctx:
        ctx.save_graph_instrument('./outs/barrier')
        opt_graph = hidet.tos.optimize(graph)


def demo_transposed_matmul():
    a = hidet.randn([10, 128, 768])
    b = hidet.randn([10, 768, 2304])

    @hidet.jit(opt=True)
    def matmul_v0(a: Tensor, b: Tensor) -> Tensor:
        return hidet.ops.matmul(a, b, algo='direct', mma='simt')

    @hidet.jit(opt=True)
    def matmul_v1(a: Tensor, b: Tensor) -> Tensor:
        return hidet.ops.matmul(a, b, algo='direct', mma='wmma_tf32_f32')

    @hidet.jit(opt=True)
    def matmul_v2(a: Tensor, b: Tensor) -> Tensor:
        return hidet.ops.matmul(a, b, algo='direct', mma='wmma_tf32_f32', ta=True)

    @hidet.jit(opt=True)
    def matmul_v3(a: Tensor, b: Tensor) -> Tensor:
        return hidet.ops.matmul(a, b, algo='direct', mma='wmma_tf32_f32', tb=True)

    @hidet.jit(opt=True)
    def matmul_v4(a: Tensor, b: Tensor) -> Tensor:
        return hidet.ops.matmul(a, b, algo='direct', mma='wmma_tf32_f32', ta=True, tb=True)

    c0 = matmul_v0(a, b)
    c1 = matmul_v1(a, b)
    # c2 = matmul_v2(a, b)
    # c3 = matmul_v3(a, b)
    # c4 = matmul_v4(a, b)

    print('et')
    print(error_tolerance(c0, c1))
    # print(error_tolerance(c0, c2))
    # print(error_tolerance(c0, c3))
    # print(error_tolerance(c0, c4))

    print('latency')
    print(matmul_v0.benchmark(a, b))
    print(matmul_v1.benchmark(a, b))
    # print(matmul_v2.benchmark(a, b))
    # print(matmul_v3.benchmark(a, b))
    # print(matmul_v4.benchmark(a, b))


def demo_debug_fuse_prologue_pass():
    m_size = 128
    n_size = 2304
    k_size = 768
    a1 = hidet.randn([1, k_size, n_size], stddev=math.sqrt(1) / math.sqrt(k_size))
    a2 = hidet.zeros([n_size])

    def func(x: Tensor):
        x = ops.matmul(x, a1)
        x = x + a2
        x = ops.strided_slice(x, starts=[0], ends=[768], axes=[2])
        x = ops.reshape(x, [8, 128, 12, 64])
        x = ops.transpose(x, [0, 2, 1, 3])
        return x
        # x = x + a2
        # x1 = ops.erf(x / 1.414) + 1.0
        # x2 = x * x1
        # x2 = x2 * 0.5
        # return x2

    @hidet.jit()
    def func_ref(x: Tensor):
        return func(x)

    @hidet.jit(opt=True, mma='wmma', parallel_k='disabled')
    def func_opt(x: Tensor):
        return func(x)

    x = hidet.randn([1, m_size, k_size])
    y1 = func_ref(x)
    hidet.space_level(2)
    y2 = func_opt(x)
    y3 = np.matmul(x.numpy(), a1.numpy()) + a2.numpy()
    print(x.shape)
    print(a1.shape)
    print('{:.3f}'.format(error_tolerance(y1, y2)))
    # print(error_tolerance(y2, y3))
    # print(error_tolerance(y1, y3))
    print(func_ref)
    print(func_opt)
    # print(y1)
    # print(y2)
    # print(y3)
    # print(y3.shape)


def demo_cumsum():
    a = hidet.array([
        [1, 2, 3, 4],
        [4, 5, 6, 7],
        [7, 8, 9, 10]
    ])
    b = hidet.ops.cumsum(a, dim=0, reverse=True, exclusive=True)
    print(a)
    print(b)


def demo_onehot():
    a = hidet.array([
        [1, 3, 2]
    ])
    b = hidet.ops.onehot(a, 4)
    print(a)
    print(b)


def demo_argmax():
    with hidet.utils.CacheDir('./outs/cache'):
        a = hidet.array([
            [0, 3, 2],
            [4, 12, 5],
            [8, 3, 9],
            [11, 2, 3]
        ])
        print(ops.argmax(a, 1))
        print(ops.argmin(a, 1))


def demo_cuda_graph():
    n = 2048
    a = hidet.randn([n, n])
    b = hidet.randn([n, n])

    @hidet.jit(opt=True)
    def func(x: Tensor, y: Tensor) -> Tensor:
        return ops.matmul(x, y)

    graph = func.flow_graph_for(a, b)
    cuda_graph_1 = graph.cuda_graph()
    cuda_graph_1.set_input_tensors([a, b])
    cuda_graph_2 = graph.cuda_graph()
    cuda_graph_2.set_input_tensors([a, b])

    stream1 = hidet.runtime.CudaStream()
    stream2 = hidet.runtime.CudaStream()
    cuda_graph_1.run(stream1)
    cuda_graph_2.run(stream2)
    stream1.synchronize()
    # cuda_graph_1.run()
    # cuda_graph_2.run()
    hidet.utils.cuda.device_synchronize()


def get_flow_graph(n=1024) -> hidet.FlowGraph:
    class Model(hidet.graph.Module):
        def __init__(self):
            super(Model, self).__init__()
            num_layers = 20
            self.linear_layers = hidet.graph.nn.Sequential([hidet.graph.nn.Linear(n, n) for _ in range(num_layers)])

        def forward(self, x: Tensor) -> Tensor:
            return self.linear_layers(x)

    model = Model()
    return model.flow_graph_for(inputs=[hidet.randn([1, n])])


def demo_pipeline():
    n = 2048
    with hidet.runtime.storage.CpuMemoryPool(max_reserve_size=5 * 1024 ** 3):
        with nvtx_annotate('init'):
            num_samples = 20
            data: List[np.ndarray] = [np.random.randn(1, n).astype('float32') for _ in range(num_samples)]
            outputs: List[np.ndarray] = []
            graph: FlowGraph = get_flow_graph(n)

        with nvtx_annotate('create graphs'):
            num_stages: int = 2
            cuda_streams: List[CudaStream] = [CudaStream() for _ in range(num_stages)]
            cuda_graphs: List[CudaGraph] = [graph.cuda_graph() for _ in range(num_stages)]

        for i in range(num_samples + num_stages - 1):
            j = i - num_stages
            if j >= 0:
                with nvtx_annotate('[{}] j_async_copy'.format(i)):
                    stream = cuda_streams[j % num_stages]
                    output_j = cuda_graphs[j % num_stages].get_output_tensors()[0]
                    output_j = output_j.cpu_async(stream)
                    stream.synchronize()
                with nvtx_annotate('[{}] j_numpy'.format(i)):
                    outputs.append(output_j.numpy(share_mem=False))
                with nvtx_annotate('[{}] j_post_processing'.format(i)):
                    time.sleep(0.0010)
            if i < num_samples:
                stream = cuda_streams[i % num_stages]
                with nvtx_annotate('[{}] i_input_preprocessing'.format(i)):
                    time.sleep(0.0010)
                with nvtx_annotate('[{}] i_array_async'.format(i)):
                    input_i = hidet.array(data[i]).cuda_async(stream)
                with nvtx_annotate('[{}] i_set_inputs_run'.format(i)):
                    cuda_graphs[i % num_stages].set_input_tensors([input_i], stream=stream)
                    cuda_graphs[i % num_stages].run(stream=stream)


if __name__ == '__main__':
    # demo_block('bert')
    # demo_parallel_k_matmul()
    # demo_batched_matmul()
    # demo_batch_matmul_precision()
    # demo_no_barrier()
    # demo_barrier()
    # demo_transposed_matmul()
    # demo_debug_fuse_prologue_pass()
    # demo_cumsum()
    # demo_onehot()
    # demo_argmax()
    # demo_cuda_graph()

    demo_pipeline()

