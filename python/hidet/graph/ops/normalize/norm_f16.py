# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import List
from hidet.ir import IRModule, dtypes
from hidet.ir.primitives import active_mask, shfl_down_sync
from hidet.ir.expr import Expr
from hidet.lang import spatial, repeat, view, cast
from hidet.lang import data_type, TensorType, i32, f16, attrs, tensor
from hidet.lang.cuda import blockIdx, threadIdx, register_tensor, syncthreads
from hidet.graph.ops.utils import Operator, Tensor, normalize_dim
from hidet.graph.ops.utils import input_like
from hidet.utils import prod
from hidet.ir import primitives as prim
from .norm import NormalizeTask


class NormalizeF16Task(NormalizeTask):
    """
    Performs the following operation in float16 precision
        mean = x.mean(dims, keep_dim=True)
        x = x - mean
        variance = square(x).mean(dims, keep_dim=True)
        x = x * rsqrt(variance + epsilon)
    """

    def implement_cuda(self, working_dir: str) -> IRModule:
        import hidet
        import math

        x, y = self.inputs[0], self.outputs[0]
        input_shape: List[Expr] = list(x.shape)
        dims = self.dims

        spatial_shape = [v for i, v in enumerate(input_shape) if i not in dims]
        reduce_shape = [int(input_shape[i]) for i in dims]
        dim_zeros = [0] * len(dims)

        reduce_extent = prod(reduce_shape)

        warp_size = 32  # TODO: improve coleased loads
        block_size = min(max(warp_size, reduce_extent), 1024)
        block_size = math.ceil(block_size / warp_size) * warp_size
        repeat_reduction = math.ceil(reduce_extent / block_size)

        task_layout = spatial(*spatial_shape)
        grid_size = task_layout.num_workers

        accumulate_dtype = data_type(self.attrs['accumulate_dtype'])

        shm_count = math.ceil(block_size / warp_size)
        used_smem_bytes_per_block = shm_count

        stages = math.ceil(math.log(block_size) / math.log(warp_size))
        assert stages <= 2

        with hidet.script_module() as module:

            @hidet.script
            def welford_combine(
                mean_a: TensorType(dtype=accumulate_dtype, shape=[1]),
                m2_a: TensorType(dtype=accumulate_dtype, shape=[1]),
                count_a: TensorType(dtype=i32, shape=[1]),
                mean_b: TensorType(dtype=accumulate_dtype, shape=[1]),
                m2_b: TensorType(dtype=accumulate_dtype, shape=[1]),
                count_b: TensorType(dtype=i32, shape=[1]),
            ):
                count = count_a[0] + count_b[0]
                if count == 0:
                    return
                delta = mean_b[0] - mean_a[0]

                mean_a[0] = mean_a[0] + delta * cast(count_b[0], accumulate_dtype) / cast(count, accumulate_dtype)
                m2_a[0] = (
                    m2_a[0]
                    + m2_b[0]
                    + delta
                    * delta
                    * cast(count_a[0], accumulate_dtype)
                    * cast(count_b[0], accumulate_dtype)
                    / cast(count, accumulate_dtype)
                )
                count_a[0] = count

            @hidet.script
            def norm_kernel(x: f16[x.shape], y: f16[y.shape]):
                attrs.cuda.grid_dim = grid_size
                attrs.cuda.block_dim = block_size
                attrs.cuda.min_blocks = 1

                # this is used for multi-level reduction
                smem_mean = tensor('shared', accumulate_dtype, shape=[used_smem_bytes_per_block])
                smem_m2 = tensor('shared', accumulate_dtype, shape=[used_smem_bytes_per_block])
                smem_count = tensor('shared', i32, shape=[used_smem_bytes_per_block])

                # cache repeated loads
                regs_repeat = tensor('register', f16, shape=[repeat_reduction])

                reg16 = register_tensor(f16, [1])
                mean_final = register_tensor(accumulate_dtype, [1])
                m2_final = register_tensor(accumulate_dtype, [1])
                count_final = register_tensor('int32', [1])

                mean_final[0] = accumulate_dtype.zero
                m2_final[0] = accumulate_dtype.zero
                count_final[0] = dtypes.int32.zero

                for spatial_idxs in task_layout.on(blockIdx.x, bind_tuple=True):
                    ele_idx = spatial_idxs + dim_zeros
                    norm_tensor = ~x[ele_idx]
                    flat_tensor = view(norm_tensor, f16[reduce_extent])

                    reduce_mapping = repeat(repeat_reduction) * spatial(block_size)
                    for reduction_idx in reduce_mapping.on(threadIdx.x):
                        mean = register_tensor(accumulate_dtype, [1])
                        m2 = register_tensor(accumulate_dtype, [1])
                        count = register_tensor('int32', [1])
                        other_mean = register_tensor(accumulate_dtype, [1])
                        other_m2 = register_tensor(accumulate_dtype, [1])
                        other_count = register_tensor('int32', [1])

                        if reduction_idx < reduce_extent:
                            reg16[0] = flat_tensor[reduction_idx]
                            count[0] = 1
                        else:
                            reg16[0] = f16.zero
                            count[0] = 0
                        regs_repeat[reduction_idx // block_size] = reg16[0]

                        mean[0] = reg16[0]
                        m2[0] = f16.zero

                        # Warp reduce by shuffle down
                        mask = active_mask()
                        other_mean[0] = shfl_down_sync(mask, mean[0], 16, 32)
                        other_m2[0] = shfl_down_sync(mask, m2[0], 16, 32)
                        other_count[0] = shfl_down_sync(mask, count[0], 16, 32)
                        welford_combine(mean, m2, count, other_mean, other_m2, other_count)

                        other_mean[0] = shfl_down_sync(mask, mean[0], 8, 32)
                        other_m2[0] = shfl_down_sync(mask, m2[0], 8, 32)
                        other_count[0] = shfl_down_sync(mask, count[0], 8, 32)
                        welford_combine(mean, m2, count, other_mean, other_m2, other_count)

                        other_mean[0] = shfl_down_sync(mask, mean[0], 4, 32)
                        other_m2[0] = shfl_down_sync(mask, m2[0], 4, 32)
                        other_count[0] = shfl_down_sync(mask, count[0], 4, 32)
                        welford_combine(mean, m2, count, other_mean, other_m2, other_count)

                        other_mean[0] = shfl_down_sync(mask, mean[0], 2, 32)
                        other_m2[0] = shfl_down_sync(mask, m2[0], 2, 32)
                        other_count[0] = shfl_down_sync(mask, count[0], 2, 32)
                        welford_combine(mean, m2, count, other_mean, other_m2, other_count)

                        other_mean[0] = shfl_down_sync(mask, mean[0], 1, 32)
                        other_m2[0] = shfl_down_sync(mask, m2[0], 1, 32)
                        other_count[0] = shfl_down_sync(mask, count[0], 1, 32)
                        welford_combine(mean, m2, count, other_mean, other_m2, other_count)

                        if stages > 1 and threadIdx.x % warp_size == 0:
                            smem_mean[threadIdx.x // warp_size] = mean[0]
                            smem_m2[threadIdx.x // warp_size] = m2[0]
                            smem_count[threadIdx.x // warp_size] = count[0]

                        syncthreads()

                        # reduce shared memory with just a single warp
                        if stages > 1 and threadIdx.x < warp_size:
                            mean[0] = smem_mean[threadIdx.x] if threadIdx.x < shm_count else accumulate_dtype.zero
                            m2[0] = smem_m2[threadIdx.x] if threadIdx.x < shm_count else accumulate_dtype.zero
                            count[0] = smem_count[threadIdx.x] if threadIdx.x < shm_count else 0

                        syncthreads()

                        if stages > 1 and threadIdx.x < warp_size:
                            other_mean[0] = shfl_down_sync(mask, mean[0], 16, 32)
                            other_m2[0] = shfl_down_sync(mask, m2[0], 16, 32)
                            other_count[0] = shfl_down_sync(mask, count[0], 16, 32)
                            welford_combine(mean, m2, count, other_mean, other_m2, other_count)

                            other_mean[0] = shfl_down_sync(mask, mean[0], 8, 32)
                            other_m2[0] = shfl_down_sync(mask, m2[0], 8, 32)
                            other_count[0] = shfl_down_sync(mask, count[0], 8, 32)
                            welford_combine(mean, m2, count, other_mean, other_m2, other_count)

                            other_mean[0] = shfl_down_sync(mask, mean[0], 4, 32)
                            other_m2[0] = shfl_down_sync(mask, m2[0], 4, 32)
                            other_count[0] = shfl_down_sync(mask, count[0], 4, 32)
                            welford_combine(mean, m2, count, other_mean, other_m2, other_count)

                            other_mean[0] = shfl_down_sync(mask, mean[0], 2, 32)
                            other_m2[0] = shfl_down_sync(mask, m2[0], 2, 32)
                            other_count[0] = shfl_down_sync(mask, count[0], 2, 32)
                            welford_combine(mean, m2, count, other_mean, other_m2, other_count)

                            other_mean[0] = shfl_down_sync(mask, mean[0], 1, 32)
                            other_m2[0] = shfl_down_sync(mask, m2[0], 1, 32)
                            other_count[0] = shfl_down_sync(mask, count[0], 1, 32)
                            welford_combine(mean, m2, count, other_mean, other_m2, other_count)

                        # at theis point mean, m2, count on T0 has the correct data for this iteration
                        # we store this to shm and let everyone else read
                        if threadIdx.x == 0:
                            smem_mean[0] = mean[0]
                            smem_m2[0] = m2[0]
                            smem_count[0] = count[0]
                        syncthreads()

                        mean[0] = smem_mean[0]
                        m2[0] = smem_m2[0]
                        count[0] = smem_count[0]
                        syncthreads()

                        # we have mean and var here for this iteration
                        welford_combine(mean_final, m2_final, count_final, mean, m2, count)

                # end of mean and var calculation, perform write back
                m2_final[0] = m2_final[0] / cast(count_final[0], accumulate_dtype)

                for spatial_idxs in task_layout.on(blockIdx.x, bind_tuple=True):
                    ele_idx = spatial_idxs + dim_zeros
                    norm_tensor = ~y[ele_idx]
                    flat_tensor = view(norm_tensor, f16[reduce_extent])

                    reduce_mapping = repeat(repeat_reduction) * spatial(block_size)
                    for reduction_idx in reduce_mapping.on(threadIdx.x):
                        if reduction_idx < reduce_extent:
                            val = regs_repeat[reduction_idx // block_size]
                            normed = (val - mean_final[0]) * prim.rsqrt(
                                m2_final[0] + cast(self.attrs['epsilon'], accumulate_dtype)
                            )
                            flat_tensor[reduction_idx] = normed

        ir_module = module.ir_module()
        return ir_module


class NormalizeF16Op(Operator):
    def __init__(self, x: Tensor, dims, epsilon: float, accumulate_dtype: str):
        rank = len(x.shape)
        dims = normalize_dim(dims, rank=rank)
        super().__init__(
            inputs=[x],
            attributes={'dims': dims, 'epsilon': epsilon, 'accumulate_dtype': accumulate_dtype},
            task=NormalizeF16Task(input_like(x, 'x'), dims, epsilon, accumulate_dtype),
        )


def normalize_f16(x: Tensor, axis: List[int], epsilon: float = 1e-5, accumulate_dtype: str = 'float32') -> Tensor:
    """Instance norm.

    Parameters
    ----------
    x: Tensor
        The data to be normalized.
    axis: int
        The axis of channel dimension.
    epsilon: float
        The epsilon added to variance.
    accumulate_dtype: str
        The precision used for accumulation during reduction

    Returns
    -------
    ret: Tensor
        The normalized tensor.
    """
    return NormalizeF16Op(x, axis, epsilon, accumulate_dtype).outputs[0]
