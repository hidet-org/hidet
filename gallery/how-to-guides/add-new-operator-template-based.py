"""
Using Template-based Scheduling
===============================

In the previous tutorial, we have learned how to define a new operator with rule-based scheduling. Rule-based scheduling
is a convenient way to define a new operator, but it is not efficient enough for operators with large amount of
reduction. In this tutorial, we will learn how to define a new operator with **template-based scheduling**.
Template-based scheduling allows us to define a tensor program template, and the template will be instantiated for
different input shapes and tunable hyper-parameters.

Override ``implement_cuda()`` method
------------------------------------
The :class:`~hidet.ir.task.Task` class have two methods :code:`implement_cpu()` and :code:`implement_cuda()` that
can be override when we define a new task.

"""
import hidet
from hidet.ir.compute import TensorNode, compute, reduce
from hidet.ir.task import Task
from hidet.ir.func import IRModule


class BatchMatmulFp16Task(Task):
    def __init__(self, a: TensorNode, b: TensorNode):
        batch_size, m_size, k_size = a.const_shape()
        batch_size, k_size, n_size = b.const_shape()
        c = compute(
            name='c',
            shape=[batch_size, m_size, n_size],
            fcompute=lambda p, i, j: reduce(
                shape=[k_size],
                fcompute=lambda k: a[p, i, k] * b[p, k, j],
                reduce_type='sum',
            ),
        )
        super().__init__(
            name='batch_matmul_fp16',
            inputs=[a, b],
            outputs=[c],
            attributes={
                'batch_size': batch_size,
                'm_size': m_size,
                'n_size': n_size,
                'k_size': k_size,
            },
        )

    def allow_epilogue(self) -> bool:
        return False

    def implement_cuda(self, working_dir: str) -> IRModule:
        # override this method to use template-based scheduling
        return batch_matmul_mma_fp16_schedule(self)


# %%
# In above task definition, we override the :code:`implement_cuda()` method to use template-based scheduling. Inside
# the :code:`implement_cuda()` method, we call the :code:`batch_matmul_mma_fp16_schedule()` function to get a tensor
# program that implements the computation defined in the task.
#
# Implement the tensor-program
# ----------------------------
# We can implement the :code:`batch_matmul_mma_fp16_schedule()` function in the following way. This function is
# complicated. To learn what it does, we should know both CUDA programming and Hidet Script. Feel free to skip it for
# now.
#
# .. note::
#   :class: margin
#
#   This function defines the tensor program based on *Hidet Script*. Hidet Script is another domain-specific language
#   in Hidet that allows developers to write tensor programs in python syntax. We will add more documentation
#   to introduce Hidet Script in the future when it gets more stable.
#


def batch_matmul_mma_fp16_schedule(task: BatchMatmulFp16Task) -> IRModule:
    from hidet.lang import f16, spatial, repeat, tensor, attr, grid, printf, cast
    from hidet.lang.mapping import repeat, spatial
    from hidet.lang.cuda import blockIdx, threadIdx, syncthreads
    from hidet.lang.cuda import MmaConfig, mma_sync

    # get the workload size
    bs = task.attrs['batch_size']
    m_size = task.attrs['m_size']
    n_size = task.attrs['n_size']
    k_size = task.attrs['k_size']

    # define the template hyper-parameters
    mma_config = MmaConfig.m16n8k8_f16_f16()
    block_m, block_n, block_k = 128, 128, 8
    warp_m, warp_n, warp_k = 64, 64, 8
    warp_count_m, warp_count_n, warp_count_k = 2, 2, 1
    mma_m, mma_n, mma_k = mma_config.m, mma_config.n, mma_config.k  # 16, 8, 8
    mma_count_m, mma_count_n, mma_count = 4, 8, 1
    threads = warp_count_m * warp_count_n * warp_count_k * 32

    # define the tensor program
    with hidet.script_module() as module:

        @hidet.script
        def load_regs_a(
            smem_a: f16[block_m, block_k], regs_a: f16[4, mma_config.a_elements]
        ):
            """Load A registers from shared memory."""
            warp_id, lane_id = threadIdx.x / 32, threadIdx.x % 32
            for wi, wj, wk in spatial(warp_count_m, warp_count_n, warp_count_k).on(
                warp_id
            ):
                for mi in range(mma_count_m):
                    p = 0
                    for i, k in mma_config.a_load_map.on(lane_id):
                        regs_a[mi, p] = smem_a[
                            wi * warp_m + mi * mma_m + i, wk * warp_k + k
                        ]
                        p += 1

        @hidet.script
        def load_regs_b(
            smem_b: f16[block_k, block_n], regs_b: f16[8, mma_config.b_elements]
        ):
            """Load B registers from shared memory."""
            warp_id, lane_id = threadIdx.x / 32, threadIdx.x % 32
            for wi, wj, wk in spatial(warp_count_m, warp_count_n, warp_count_k).on(
                warp_id
            ):
                for mj in range(mma_count_n):
                    p = 0
                    for k, j in mma_config.b_load_map.on(lane_id):
                        regs_b[mj, p] = smem_b[
                            wk * warp_k + k, wj * warp_n + mj * mma_n + j
                        ]
                        p += 1

        @hidet.script
        def warp_mma(
            regs_a: f16[4, mma_config.a_elements],
            regs_b: f16[8, mma_config.b_elements],
            regs_c: f16[4, 8, mma_config.c_elements],
        ):
            """Perform warp-level matrix multiplication."""
            for mi, mj in repeat(mma_count_m, mma_count_n).on(0):
                mma_sync(mma_config, ~regs_a[mi, 0], ~regs_b[mj, 0], ~regs_c[mi, mj, 0])

        @hidet.script
        def store_c(regs_c: f16[4, 8, mma_config.c_elements], c: f16[bs, m_size, n_size]):
            """Store C registers to global memory."""
            warp_id, lane_id = threadIdx.x / 32, threadIdx.x % 32
            offset_m, offset_n = blockIdx.x * block_m, blockIdx.y * block_n
            gmem_c = c[blockIdx.z, offset_m:, offset_n:]
            for k_round in range(warp_count_k):
                for wi, wj, wk in spatial(warp_count_m, warp_count_n, warp_count_k).on(
                    warp_id
                ):
                    if wk == k_round:
                        for mi, mj in repeat(mma_count_m, mma_count_n).on(0):
                            p = 0
                            for i, j in mma_config.c_store_map.on(lane_id):
                                gmem_c.write(
                                    [
                                        wi * warp_m + mi * mma_m + i,
                                        wj * warp_n + mj * mma_n + j,
                                    ],
                                    regs_c[mi, mj, p],
                                    protected=True,
                                )
                                p += 1

        @hidet.script
        def batch_matmul_kernel(
            a: f16[bs, m_size, k_size],
            b: f16[bs, k_size, n_size],
            c: f16[bs, m_size, n_size],
        ):
            """Batch matrix multiplication kernel."""
            attr.cuda_grid_dim = (
                (m_size + block_m - 1) // block_m,
                (n_size + block_n - 1) // block_n,
                bs,
            )
            attr.cuda_block_dim = threads
            offset_m, offset_n = blockIdx.x * block_m, blockIdx.y * block_n
            smem_a = tensor('shared', 'float16', [block_m, block_k])
            smem_b = tensor('shared', 'float16', [block_k, block_n])
            regs_a = tensor('register', 'float16', [4, mma_config.a_elements])
            regs_b = tensor('register', 'float16', [8, mma_config.b_elements])
            regs_c = tensor('register', 'float16', [4, 8, mma_config.c_elements])

            for i, j, p in grid(4, 8, mma_config.c_elements):
                regs_c[i, j, p] = 0.0

            for k0 in range((k_size + block_k - 1) // block_k):
                offset_k = k0 * block_k
                gmem_a = a[blockIdx.z, offset_m:, offset_k:]
                gmem_b = b[blockIdx.z, offset_k:, offset_n:]
                for i, k in repeat(8, 1).spatial(16, 8).on(threadIdx.x):
                    smem_a[i, k] = gmem_a.read([i, k], protected=True)
                for k, j in repeat(8, 1).spatial(1, 128).on(threadIdx.x):
                    smem_b[k, j] = gmem_b.read([k, j], protected=True)
                syncthreads()
                load_regs_a(smem_a, regs_a)
                load_regs_b(smem_b, regs_b)
                warp_mma(regs_a, regs_b, regs_c)
                syncthreads()
            store_c(regs_c, c)

    ir_module = module.ir_module()
    return ir_module


# %%
# Define the operator
# -------------------
# The remaining part is the same as the rule-based scheduling method to add new operator.
from hidet.graph import Operator, Tensor
from hidet.graph.ops.definitions.utils import input_like


class BatchMatmulFp16Op(Operator):
    def __init__(self, a: Tensor, b: Tensor):
        assert a.dtype == hidet.float16 and b.dtype == hidet.float16
        super().__init__(
            inputs=[a, b],
            attributes={},
            task=BatchMatmulFp16Task(input_like(a, 'a'), input_like(b, 'b')),
        )


def batch_matmul_fp16(a: Tensor, b: Tensor) -> Tensor:
    return BatchMatmulFp16Op(a, b).get_output(0)


def demo_usage():
    a = hidet.randn([1, 2, 2], dtype='float16', device='cuda')
    b = hidet.randn([1, 2, 2], dtype='float16', device='cuda')
    c = batch_matmul_fp16(a, b)
    print(a)
    print(b)
    print(c)


demo_usage()

# %%
# Generated Source Code
# ---------------------
# If you are interested in the generated source code, here it is:

# sphinx_gallery_start_ignore
a = hidet.randn([1, 2, 2], dtype='float16', device='cuda')
b = hidet.randn([1, 2, 2], dtype='float16', device='cuda')
op = BatchMatmulFp16Op(a, b)
c = op.get_output(0)
func = op.task_func
import os

relative_path = os.path.relpath(
    func.src_path, os.path.dirname(hidet.utils.hidet_cache_dir())
)
source_path = func.src_path
# sphinx_gallery_end_ignore

# we hide the code to get the source path for simplicity
print('Generated source path (relative to hidet cache root): \n{}'.format(relative_path))
print()
print('Generated source code:')
with open(source_path, 'r') as f:
    print(f.read())

# %%
# Summary
# -------
# In this tutorial, we have shown how to use the template-based scheduling mechanism to add new operators. Basically,
# what we need to do is to override the **implement_cuda** or **implement_cpu** method of the task class, and implement
# the task to get an IR module. In this example, we used Hidet Script to implement the task, but you can also use
# other ways such as IR builder.
