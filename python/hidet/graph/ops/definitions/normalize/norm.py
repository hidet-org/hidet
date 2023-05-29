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
from hidet.ir import IRModule
from hidet.ir.primitives import active_mask, shfl_down_sync
from hidet.ir.compute import reduce
from hidet.lang import spatial, repeat, view, cast
from hidet.lang import data_type, TensorType, i32, f32, attrs, tensor
from hidet.lang.cuda import blockIdx, threadIdx, register_tensor, syncthreads
from hidet.graph.ops.definitions.utils import Task, Operator, Tensor, TensorNode
from hidet.graph.ops.definitions.utils import compute, input_like, normalize_dim
from hidet.utils import prod
from hidet.ir import primitives as prim
from hidet.ir.expr import convert


class NormalizeTask(Task):
    """
    Performs the following operation
        mean = x.mean(dims, keep_dim=True)
        x = x - mean
        variance = square(x).mean(dims, keep_dim=True)
        x = x * rsqrt(variance + epsilon)
    """

    def __init__(self, x: TensorNode, dims: List[int], epsilon: float, accumulate_dtype: str):
        dtype_str = x.ttype.dtype.name
        x_shape = x.const_shape
        reduce_shape = []
        other_shape = []
        for idx, size in enumerate(x_shape):
            if idx in dims:
                reduce_shape.append(size)
            else:
                other_shape.append(size)
        epsilon = convert(epsilon, dtype=dtype_str)

        def mean_compute(*indices):
            def mean_reduce(*reduction_axis):
                x_indices = []
                p = 0
                q = 0
                for i in range(len(x.shape)):
                    if i not in dims:
                        x_indices.append(indices[p])
                        p += 1
                    else:
                        x_indices.append(reduction_axis[q])
                        q += 1
                assert p == len(indices) and q == len(reduction_axis)
                return x[x_indices]

            return reduce(
                shape=reduce_shape, fcompute=mean_reduce, reduce_type='avg', accumulate_dtype=accumulate_dtype
            )

        mean = compute(name='mean', shape=other_shape, fcompute=mean_compute)

        def var_compute(*indices):
            def var_reduce(*reduction_axis):
                x_indices = []
                p = 0
                q = 0
                for i in range(len(x.shape)):
                    if i not in dims:
                        x_indices.append(indices[p])
                        p += 1
                    else:
                        x_indices.append(reduction_axis[q])
                        q += 1
                assert p == len(indices) and q == len(reduction_axis)
                return prim.pow(x[x_indices] - mean[indices], 2)

            return reduce(shape=reduce_shape, fcompute=var_reduce, reduce_type='avg', accumulate_dtype=accumulate_dtype)

        var = compute(name='var', shape=other_shape, fcompute=var_compute)

        def norm_compute(*indices):
            mean_var_indices = [index for id, index in enumerate(indices) if id not in dims]
            return (x[indices] - mean[mean_var_indices]) * prim.rsqrt(var[mean_var_indices] + epsilon)

        y = compute(name='y', shape=x_shape, fcompute=norm_compute)

        self.dims: List[int] = dims

        super().__init__(
            name='normalize_{}'.format(dtype_str),
            inputs=[x],
            outputs=[y],
            attributes={'dims': dims, 'accumulate_dtype': accumulate_dtype, 'epsilon': epsilon},
        )

    def allow_prologue(self) -> bool:
        return False

    def allow_epilogue(self) -> bool:
        return False

    def implement_cuda(self, working_dir: str) -> IRModule:
        import hidet
        import math

        x, y = self.inputs[0], self.outputs[0]
        input_shape: List[int] = list(x.const_shape)
        dims = self.dims

        spatial_shape = [v for i, v in enumerate(input_shape) if i not in dims]
        reduce_shape = [input_shape[i] for i in dims]
        dim_zeros = [0] * len(dims)

        reduce_extent = prod(reduce_shape)

        warp_size = 32
        block_size = min(max(warp_size, reduce_extent), 1024)
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

                mean_a[0] = mean_a[0] + delta * cast(count_b[0], f32) / cast(count, f32)
                m2_a[0] = (
                    m2_a[0] + m2_b[0] + delta * delta * cast(count_a[0], f32) * cast(count_b[0], f32) / cast(count, f32)
                )
                count_a[0] = count

            @hidet.script
            def norm_kernel(x: f32[x.const_shape], y: f32[y.const_shape]):
                attrs.cuda.grid_dim = grid_size
                attrs.cuda.block_dim = block_size
                attrs.cuda.min_blocks = 1

                # this is used for multi-level reduction
                smem_mean = tensor('shared', accumulate_dtype, shape=[used_smem_bytes_per_block])
                smem_m2 = tensor('shared', accumulate_dtype, shape=[used_smem_bytes_per_block])
                smem_count = tensor('shared', i32, shape=[used_smem_bytes_per_block])

                # cache repeated loads
                regs_repeat = tensor('register', f32, shape=[repeat_reduction])

                reg32 = register_tensor(f32, [1])
                mean_final = register_tensor(accumulate_dtype, [1])
                m2_final = register_tensor(accumulate_dtype, [1])
                count_final = register_tensor('int32', [1])

                mean_final[0] = accumulate_dtype.zero
                m2_final[0] = accumulate_dtype.zero
                count_final[0] = i32.zero

                for spatial_idxs in task_layout.on(blockIdx.x, bind_tuple=True):
                    # note, this is evaluated at compile time
                    ele_idx = spatial_idxs + dim_zeros
                    norm_tensor = ~x[ele_idx]
                    flat_tensor = view(norm_tensor, f32[reduce_extent])

                    reduce_mapping = repeat(repeat_reduction) * spatial(block_size)
                    for reduction_idx in reduce_mapping.on(threadIdx.x):
                        mean = register_tensor(accumulate_dtype, [1])
                        m2 = register_tensor(accumulate_dtype, [1])
                        count = register_tensor('int32', [1])
                        other_mean = register_tensor(accumulate_dtype, [1])
                        other_m2 = register_tensor(accumulate_dtype, [1])
                        other_count = register_tensor('int32', [1])

                        if reduction_idx < reduce_extent:
                            reg32[0] = flat_tensor[reduction_idx]
                            count[0] = 1
                        else:
                            reg32[0] = f32.zero
                            count[0] = 0
                        regs_repeat[reduction_idx // block_size] = reg32[0]

                        mean[0] = reg32[0]
                        m2[0] = f32.zero

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
                            mean[0] = smem_mean[threadIdx.x] if threadIdx.x < shm_count else f32.zero
                            m2[0] = smem_m2[threadIdx.x] if threadIdx.x < shm_count else f32.zero
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
                m2_final[0] = m2_final[0] / cast(count_final[0], f32)

                for spatial_idxs in task_layout.on(blockIdx.x, bind_tuple=True):
                    ele_idx = spatial_idxs + dim_zeros
                    norm_tensor = ~y[ele_idx]
                    flat_tensor = view(norm_tensor, f32[reduce_extent])

                    reduce_mapping = repeat(repeat_reduction) * spatial(block_size)
                    for reduction_idx in reduce_mapping.on(threadIdx.x):
                        if reduction_idx < reduce_extent:
                            val = regs_repeat[reduction_idx // block_size]
                            normed = (val - mean_final[0]) * prim.rsqrt(m2_final[0] + self.attrs['epsilon'])
                            flat_tensor[reduction_idx] = normed

        ir_module = module.ir_module()
        return ir_module


class NormalizeOp(Operator):
    def __init__(self, x: Tensor, dims, epsilon: float, accumulate_dtype: str):
        rank = len(x.shape)
        dims = normalize_dim(dims, rank=rank)
        super().__init__(
            inputs=[x],
            attributes={'dims': dims, 'epsilon': epsilon, 'accumulate_dtype': accumulate_dtype},
            task=NormalizeTask(input_like(x, 'x'), dims, epsilon, accumulate_dtype),
        )


def normalize(x: Tensor, axis: List[int], epsilon: float = 1e-5, accumulate_dtype: str = 'float32') -> Tensor:
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
    return NormalizeOp(x, axis, epsilon, accumulate_dtype).get_output(0)
