from typing import Union, List
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
# hidet.utils.clear_op_cache()
hidet.option.save_lower_ir()


class MatmulTask(Task):
    def __init__(self, a: TensorInput, b: TensorInput):
        c = cops.matmul(a, b, allow_1d=True)
        super().__init__(name='matmul', inputs=[a, b], outputs=[c])

        self._assert(a.shape[-1] % 2 == 0, 'k_size must be even')
        self._assert(b.shape[-1] % 2 == 0, 'n_size must be even')

    def allow_prologue(self) -> bool:
        return False

    def allow_epilogue(self) -> bool:
        return True

    def implement_cuda(self, working_dir: str) -> Union[IRModule, List[IRModule]]:
        return tune.extract_ir_modules(self.schedule_matmul_mma)

    @tune.space(
        2,
        matmul_simt_space=matmul_simt.tuning_space.iterate_space(2)
    )
    def schedule_matmul_mma(self, matmul_simt_space):
        from hidet.lang import attrs
        from hidet.lang.cuda import threadIdx, blockIdx, blockDim, gridDim
        from hidet.lang.mapping import spatial, repeat
        from hidet.lang import printf
        from hidet.ir.primitives.runtime import request_cuda_workspace

        b_size, m_size, k_size = self.inputs[0].shape
        b_size, k_size, n_size = self.inputs[1].shape
        parts = 4

        with hidet.script_module() as script_module:

            @hidet.script
            def matmul_kernel(
                a: f16[b_size, m_size, k_size],
                b: f16[b_size, k_size, n_size],
                c: f16[parts, b_size, m_size, n_size]
            ):
                attrs.func_kind = 'cuda_kernel'
                attrs.cuda.block_dim = (16, 16)
                attrs.cuda.grid_dim = ((m_size + 15) // 16, (n_size + 15) // 16, b_size * parts)

                i = blockIdx.x * blockDim.x + threadIdx.x
                j = blockIdx.y * blockDim.y + threadIdx.y
                part_size = (k_size + parts - 1) // parts

                if i < m_size and j < n_size:
                    for part, batch in spatial(parts, b_size).on(blockIdx.z):
                        acc = f16(0.0)
                        for k_inner in range(part_size):
                            k = part * part_size + k_inner
                            if k < k_size:
                                acc += a[batch, i, k] * b[batch, k, j]
                        c[part, batch, i, j] = acc

            @hidet.script
            def reduce_kernel(
                c_in: f16[parts, b_size, m_size, n_size],
                c_out: f16[b_size, m_size, n_size]
            ):
                attrs.func_kind = 'cuda_kernel'
                attrs.cuda.block_dim = 512
                attrs.cuda.grid_dim = (m_size * n_size * b_size + 511) // 512

                w = threadIdx.x + blockIdx.x * blockDim.x
                if w < m_size * n_size * b_size:
                    for b, i, j in spatial(b_size, m_size, n_size).on(w):
                        acc = f16(0.0)
                        for part in range(parts):
                            acc += c_in[part, b, i, j]
                        c_out[b, i, j] = acc

            @hidet.script
            def launch(
                a: f16[b_size, m_size, k_size],
                b: f16[b_size, k_size, n_size],
                c: f16[b_size, m_size, n_size]
            ):
                attrs.func_kind = 'public'
                matmul_simt(a, b, c, **matmul_simt_space)

        return script_module.ir_module()


class MatmulOp(Operator):
    def __init__(self, a: Tensor, b: Tensor):
        task = MatmulTask(input_like(a, 'a'), input_like(b, 'b'))
        super().__init__(inputs=[a, b], attributes={}, task=task)


def my_matmul(a: Tensor, b: Tensor) -> Tensor:
    return MatmulOp(a, b).outputs[0]


def benchmark():

    # for dtype in ['float32', 'float16']:
    for dtype in ['float16']:

        a = hidet.symbol(['b', 'm', 'k'], dtype=dtype, device='cuda')
        b = hidet.symbol(['b', 'k', 'n'], dtype=dtype, device='cuda')
        c = my_matmul(a, b)
        graph = hidet.trace_from(c, [a, b])
        cgraph = graph.build(space=2)

        for m_size, n_size, k_size in [(1024, 1024, 1024), (1024, 3072, 768), (1024, 768, 3072)]:
            # aa = hidet.randn([1, m_size, k_size], dtype=dtype, device='cuda', stddev=0.1)
            # bb = hidet.randn([1, k_size, n_size], dtype=dtype, device='cuda', stddev=0.1)
            aa = hidet.ones([1, m_size, k_size], dtype=dtype, device='cuda')
            bb = hidet.ones([1, k_size, n_size], dtype=dtype, device='cuda')
            at = aa.torch().clone()
            bt = bb.torch().clone()

            # check correctness
            # print(aa)
            # print(bb)
            c1 = cgraph(aa, bb)
            c2 = aa.torch() @ bb.torch()
            # print(c1)
            # print(c2)
            # print(c1[0, :128, :128])
            # print(c1[0, 128:, 128:])
            hidet.utils.assert_close(c1, c2, rtol=1e-2 * (k_size / 100), atol=1e-2 * (k_size / 100))

            # benchmark
            torch_latency = hidet.utils.benchmark_func(lambda: at @ bt, number=100)
            hidet_latency = hidet.utils.benchmark_func(lambda: cgraph(aa, bb), number=100)
            print(' {:4} x {:4} x {:4} torch: {:.3f}'.format(m_size, n_size, k_size, torch_latency))
            print(' {:4} x {:4} x {:4} hidet: {:.3f}'.format(m_size, n_size, k_size, hidet_latency))


def main():
    benchmark()


if __name__ == '__main__':
    main()
