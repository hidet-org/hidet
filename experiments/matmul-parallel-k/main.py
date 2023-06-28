from typing import Union, List
import numpy
import torch
import hidet
from hidet.ir.dtypes import f16, f32
from hidet.ir.type import tensor_pointer_type
from hidet.ir.expr import cast
from hidet.ir.compute import TensorNode, TensorInput, cops
from hidet.ir.stmt import launch_kernel
from hidet.ir.task import Task
from hidet.ir.module import IRModule
from hidet.graph.tensor import Tensor
from hidet.graph.operator import Operator
from hidet.graph.ops.utils import input_like
from hidet import ops
from hidet.ir.library import tune
from hidet.ir.library.cuda import matmul_simt

hidet.option.cache_dir('./outs/cache')
# hidet.utils.clear_cache_dir()
hidet.option.save_lower_ir()


class MatmulTask(Task):
    def __init__(self, a: TensorInput, b: TensorInput):
        c = cops.matmul(a, b, allow_1d=True)
        super().__init__(name='matmul', inputs=[a, b], outputs=[c])

        self._assert(a.shape[-1] % 2 == 0, 'k_size must be even')
        self._assert(b.shape[-1] % 2 == 0, 'n_size must be even')
        self.dtype = a.type.dtype

    def allow_prologue(self) -> bool:
        return False

    def allow_epilogue(self) -> bool:
        return True

    def implement_cuda(self, working_dir: str) -> Union[IRModule, List[IRModule]]:
        return tune.extract_ir_modules(self.schedule_matmul_mma)

    @tune.space(
        2,
        parallel_k=[1, 2, 4],
        block_shape=[
            (64, 64, 8),
            (48, 128, 8),
            (128, 48, 8),
            (128, 128, 8),
            (64, 128, 8),
            (128, 64, 8),

            (64, 64, 16),
            (48, 128, 16),
            (128, 48, 16),
            (64, 128, 16),
            (128, 64, 16),
            (128, 128, 16),
        ],
        warp_shape=[
            (64, 16, 8),
            (48, 64, 8),
            (64, 48, 8),
            (16, 64, 8),
            (64, 32, 8),
            (32, 64, 8),
            (32, 32, 8),
            (64, 64, 8),
            (64, 16, 16),
            (16, 64, 16),
            (64, 32, 16),
            (32, 64, 16),
            (32, 32, 16),
            (64, 64, 16),
        ],
        warp_threads=[(4, 8)],
        thread_shape=[(4, 4)],
        swizzle_tile=[1],
        arch=['sm_70']
    )
    @tune.space(
        1,
        parallel_k=[1],
        block_shape=[(64, 128, 8)],
        warp_shape=[(32, 64, 8)],
        warp_threads=[(4, 8)],
        thread_shape=[(4, 4)],
        # parallel_k=[8],
        # block_shape=[(128, 64, 16)],
        # warp_shape=[(32, 32, 16)],
        # warp_threads=[(4, 8)],
        # thread_shape=[(4, 4)],
        swizzle_tile=[1],
        arch=['sm_70'],
        # combinations={
        #     "block_shape, warp_shape, warp_threads, thread_shape": [
        #         [(64, 64, 16), (32, 64, 8), (4, 8), (4, 4)],
        #     ]
        # }
    )
    def schedule_matmul_mma(
        self,
        parallel_k=1, block_shape=(64, 64, 16), warp_shape=(32, 64, 8), warp_threads=(4, 8), thread_shape=(4, 4),
        swizzle_tile=1, arch='sm_70'
    ):
        from hidet.lang import attrs

        b_size, m_size, k_size = self.inputs[0].shape
        b_size, k_size, n_size = self.inputs[1].shape
        dtype = self.dtype

        with hidet.script_module() as script_module:
            @hidet.script
            def launch(
                a: dtype[b_size, m_size, k_size],
                b: dtype[b_size, k_size, n_size],
                c: dtype[b_size, m_size, n_size]
            ):
                attrs.func_kind = 'public'
                matmul_simt(
                    a, b, c,
                    parallel_k=parallel_k, block_shape=block_shape, warp_shape=warp_shape,
                    warp_threads=warp_threads, thread_shape=thread_shape, swizzle_tile=swizzle_tile, arch=arch
                )

        return script_module.ir_module()


class MatmulOp(Operator):
    def __init__(self, a: Tensor, b: Tensor):
        task = MatmulTask(input_like(a, 'a'), input_like(b, 'b'))
        super().__init__(inputs=[a, b], attributes={}, task=task)


def my_matmul(a: Tensor, b: Tensor) -> Tensor:
    return MatmulOp(a, b).outputs[0]


def benchmark():
    space = 2

    for dtype in ['float32']:

        a = hidet.symbol(['b', 'm', 'k'], dtype=dtype, device='cuda')
        b = hidet.symbol(['b', 'k', 'n'], dtype=dtype, device='cuda')
        c = my_matmul(a, b)
        graph = hidet.trace_from(c, [a, b])
        cgraph = graph.build(space=space)

        for m_size, n_size, k_size in [
            (1024, 1024, 1024),
            (512, 3072, 768),
            (512, 768, 3072),
            (1024, 3072, 768),
            (1024, 768, 3072)
        ]:
            aa = hidet.randn([1, m_size, k_size], dtype=dtype, device='cuda', stddev=0.1 if dtype == 'float16' else 1)
            bb = hidet.randn([1, k_size, n_size], dtype=dtype, device='cuda', stddev=0.1 if dtype == 'float16' else 1)

            c1 = cgraph(aa, bb)
            hidet.cuda.synchronize()

            c2 = aa.torch() @ bb.torch()
            hidet.cuda.synchronize()

            sa = hidet.symbol_like(aa)
            sb = hidet.symbol_like(bb)
            graph2 = hidet.trace_from(hidet.ops.batch_matmul(sa, sb), [sa, sb])
            cgraph2 = graph2.build(space=space)
            c3 = cgraph2(aa, bb)
            hidet.cuda.synchronize()

            tol = 1e-3 if dtype == 'float32' else 1e-2
            hidet.utils.assert_close(c1, c2, rtol=tol, atol=tol)
            hidet.utils.assert_close(c1, c3, rtol=tol, atol=tol)

            # benchmark
            at = aa.torch()
            bt = bb.torch()
            ct = at @ bt
            torch_latency = hidet.utils.benchmark_func(lambda: torch.matmul(at, bt, out=ct), number=100, repeat=10)
            hidet_latency = hidet.utils.benchmark_func(lambda: cgraph(aa, bb), number=100, repeat=10)
            hidet_latency2 = hidet.utils.benchmark_func(lambda: cgraph2(aa, bb), number=100, repeat=10)
            print(' {:4} x {:4} x {:4}     torch: {:.3f}'.format(m_size, n_size, k_size, torch_latency))
            print(' {:4} x {:4} x {:4}     hidet: {:.3f} {:.2f}'.format(m_size, n_size, k_size, hidet_latency, hidet_latency / torch_latency))
            print(' {:4} x {:4} x {:4} hidet (s): {:.3f} {:.2f}'.format(m_size, n_size, k_size, hidet_latency2, hidet_latency2 / torch_latency))


def main():
    benchmark()


if __name__ == '__main__':
    main()
