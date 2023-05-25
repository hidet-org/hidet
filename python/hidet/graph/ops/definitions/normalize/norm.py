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
from typing import List, Union, Optional, Sequence
from hidet.ir import IRModule, dtypes
from hidet.ir.primitives import active_mask, shfl_down_sync, shfl_sync
from hidet.ir.compute import ReduceOperation, reduce
from hidet.ir.type import data_type
from hidet.ir.layout import DataLayout
from hidet.lang import f16, f32, spatial, repeat, attr, tensor_pointer
from hidet.lang.cuda import blockIdx, threadIdx, register_tensor
from hidet.graph.ops.definitions.utils import Task, Operator, Tensor, TensorNode, ReduceType
from hidet.graph.ops.definitions.utils import compute, input_like, normalize_dim
from hidet.graph.ops.definitions.arithmetic import square, rsqrt
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
            attributes={'dims': dims, 'accumulate_dtype': accumulate_dtype},
        )

    def allow_prologue(self) -> bool:
        return False

    def allow_epilogue(self) -> bool:
        return True

    def implement_cuda(self, working_dir: str) -> IRModule:
        import hidet
        x, y = self.inputs[0], self.outputs[0]
        shape: List[int] = list(x.const_shape)
        dims = self.dims
        spaitial_shape = [v for i, v in enumerate(shape) if i not in dims]
        reduce_shape = [shape[i] for i in dims]

        spatial_extent = prod(spatial_shape)
        reduce_extent = prod(reduce_shape)
        row_major = DataLayout.row_major

        warp_size = 32
        # at least warp_size, at most 1024
        block_size = min(max(warp_size, reduce_extent), 1024)

        spatial_layout = spatial(*remain_shape)
        reduction_layout = row_major(reduce_shape)

        # start with block level assignment
        spatial_layout = spatial(*(spatial_shape + ([1] * len(reduce_shape))))

        # We can only reduce 1024 numbers in a threadblock at once
        repeat_reduction = math.ceil(reduce_extent  / block_size)
        reduction_layout = repeat(repeat_reduction, 1) * spatial(block_size)


        task_layout = repeat(*repeat_shape) * spatial(*spatial_shape)
        grid_size = spatial_layout.num_workers
        accumulate_dtype = self.attrs['accumulate_dtype']

        dtype = self.inputs[0].type.dtype
        used_smem_bytes_per_block = match.ceil(block_size / warp_size)  * dtype.nbytes
        smem_mean = tensor('shared', 'int8', shape=[used_smem_bytes_per_block])
        smem_var = tensor('shared', 'int8', shape=[used_smem_bytes_per_block])

        with hidet.script_module() as module:

            @hidet.script
            def welford_combine(mean_a, m2_a, count_a, mean_b, m2_b, count_b):
                count = count_a + count_b
                delta = mean_b - mean_a
                mean = mean_a + delta * count_b / count
                m2 = m2_a + m2_b + delta * delta * count_a * count_b / count

            @hidet.script
            def norm_kernel(x: f16[x.const_shape], y: f16[y.const_shape]):
                attr.cuda_grid_dim = grid_size
                attr.cuda_block_dim = block_size
                attr.cuda_min_blocks = 1

                x_f32 = tensor_pointer('float32', shape=shape_32bit)
                x_f32 = x

                reg32 = register_tensor(f32, [1])
                mean = register_tensor(accumulate_dtype, [1])
                m2 = register_tensor(accumulate_dtype, [1])
                mean = data_type(accumulate_dtype).zero
                m2 = data_type(accumulate_dtype).zero
                count = data_type('int32').zero

                for indices in task_layout.on(threadIdx.x + blockIdx.x * block_size):
                    if x_f32_layout.within_bound(indices):
                        reg32[0] = x_f32.read(indices, protected=False)

                # Warp reduce by shuffle down
                mask = active_mask()
                rv[0] = ro.combine(rv[0], shfl_down_sync(mask, rv[0], 16, 32))
                rv[0] = ro.combine(rv[0], shfl_down_sync(mask, rv[0], 8, 32))
                rv[0] = ro.combine(rv[0], shfl_down_sync(mask, rv[0], 4, 32))
                rv[0] = ro.combine(rv[0], shfl_down_sync(mask, rv[0], 2, 32))
                rv[0] = ro.combine(rv[0], shfl_down_sync(mask, rv[0], 1, 32))
                rv[0] = shfl_sync(mask, rv[0], 0, 32)
                rv[0] = ro.finalize(acc=rv[0], size=reduce_extent)
                if threadIdx.x == 0:
                    for indices in remain_layout.on(blockIdx.x):
                        y.write(indices, rv[0], protected=False)

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
