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
from hidet.transforms.tools import add_packed_func, fuse_and_pack
from hidet.graph.ops.definitions.utils import Task, Operator, Tensor, TensorNode, ReduceType
from hidet.graph.ops.definitions.utils import compute, input_like, normalize_dim
from hidet.utils import prod


class ReduceF16Task(Task):
    def __init__(
        self, x: TensorNode, dims: List[int], keep_dim: bool, reduce_type: ReduceType, accumulate_dtype: str = 'float32'
    ):

        x_shape = x.const_shape()
        y_shape = []
        for i in range(len(x_shape)):
            if i in dims:
                if keep_dim:
                    y_shape.append(1)
            else:
                y_shape.append(x_shape[i])

        def fcompute(*indices):
            def reduce_fcompute(*reduce_indices):
                x_indices = []
                p = 0
                q = 0
                for i in range(len(x_shape)):
                    if i not in dims:
                        x_indices.append(indices[p])
                        p += 1
                    else:
                        x_indices.append(reduce_indices[q])
                        q += 1
                        if keep_dim:
                            p += 1
                assert p == len(indices) and q == len(reduce_indices)
                return x[x_indices]

            reduce_shape = [x_shape[i] for i in dims]
            return reduce(
                shape=reduce_shape, fcompute=reduce_fcompute, reduce_type=reduce_type, accumulate_dtype=accumulate_dtype
            )

        y = compute(name='y', shape=y_shape, fcompute=fcompute)

        self.dims: List[int] = dims
        self.keep_dim: bool = keep_dim
        self.reduce_type: ReduceType = reduce_type

        super().__init__(
            name='reduce_{}_f16'.format(reduce_type),
            inputs=[x],
            outputs=[y],
            attributes={
                'dims': dims,
                'keep_dim': keep_dim,
                'reduce_type': reduce_type,
                'accumulate_dtype': accumulate_dtype,
            },
        )

    def allow_prologue(self) -> bool:
        return False

    def allow_epilogue(self) -> bool:
        # return False
        rank = len(self.inputs[0].const_shape())
        if rank - 1 in self.dims:  # pylint: disable=simplifiable-if-statement
            # use self.cuda_schedule_reduce_by_warp
            return True
        else:
            # use self.cuda_schedule_reduce_by_default
            return False

    def implement_cuda(self, workding_dir: str) -> IRModule:
        rank = len(self.inputs[0].const_shape())
        if rank - 1 in self.dims:
            return self.cuda_schedule_reduce_by_warp()
        else:
            return self.cuda_schedule_reduce_by_default()

    def cuda_schedule_reduce_by_warp(self) -> IRModule:
        import hidet

        row_major = DataLayout.row_major

        warp_size = 32
        block_size = warp_size
        x, y = self.inputs[0], self.outputs[0]
        shape: List[int] = x.const_shape()
        dims = self.dims
        if self.keep_dim:
            remain_shape = [v if i not in dims else 1 for i, v in enumerate(shape)]
        else:
            remain_shape = [v for i, v in enumerate(shape) if i not in dims]
        reduce_shape = [shape[i] for i in dims]
        reduce_extent = prod(reduce_shape)
        shape_32bit = [s // 2 if i == len(shape) - 1 else s for i, s in enumerate(shape)]
        remain_layout = spatial(*remain_shape)
        x_f32_layout = row_major(shape_32bit)

        spatial_shape = []
        repeat_shape = []
        for i in range(len(shape_32bit)):
            if i == len(shape_32bit) - 1:
                spatial_shape.append(warp_size)
                repeat_shape.append((shape_32bit[i] + warp_size - 1) // warp_size)  # num warps per row
            elif i in dims:
                spatial_shape.append(1)
                repeat_shape.append(shape_32bit[i])
            else:
                spatial_shape.append(shape_32bit[i])
                repeat_shape.append(1)
        task_layout = repeat(*repeat_shape) * spatial(*spatial_shape)
        grid_size = remain_layout.num_workers
        accumulate_dtype = self.attributes['accumulate_dtype']
        reduce_type = self.attributes['reduce_type']
        ro = ReduceOperation.from_name(reduce_type)

        with hidet.script_module() as module:

            @hidet.script
            def reduce_kernel(x: f16[x.const_shape()], y: f16[y.const_shape()]):
                attr.cuda_grid_dim = grid_size
                attr.cuda_block_dim = block_size
                attr.cuda_min_blocks = 1

                x_f32 = tensor_pointer('float32', shape=shape_32bit)
                x_f32 = x

                reg32 = register_tensor(f32, [1])
                regs16 = tensor_pointer('float16', shape=[2])
                regs16 = reg32
                rv = register_tensor(accumulate_dtype, [1])
                rv[0] = ro.initial_value(data_type(accumulate_dtype))
                for indices in task_layout.on(threadIdx.x + blockIdx.x * block_size):
                    if x_f32_layout.within_bound(indices):
                        reg32[0] = x_f32.read(indices, protected=False)
                        rv[0] = ro.combine(rv[0], regs16[0])
                        rv[0] = ro.combine(rv[0], regs16[1])
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
        fuse_and_pack(ir_module, reduce_kernel, task=self)
        return ir_module

    def cuda_schedule_reduce_by_default(self) -> IRModule:
        import hidet

        x, y = self.inputs[0], self.outputs[0]
        dims = self.dims
        shape: List[int] = x.const_shape()

        if self.keep_dim:
            remain_shape = [v if i not in dims else 1 for i, v in enumerate(shape)]
        else:
            remain_shape = [v for i, v in enumerate(shape) if i not in dims]
        shape_32bit = [s // 2 if i == len(shape) - 1 else s for i, s in enumerate(shape)]
        # In this schedule, remain_shape[-1] == shape[-1], as the last dim is not reduced
        remain_shape_32bit = [s // 2 if i == len(remain_shape) - 1 else s for i, s in enumerate(remain_shape)]
        remain_extent = prod(remain_shape_32bit)
        reduce_shape = [shape[i] for i in dims]
        reduce_extent = prod(reduce_shape)

        block_size = min(256, remain_extent)
        remain_layout = spatial(*remain_shape_32bit)

        spatial_shape = []
        repeat_shape = []
        for i in range(len(shape_32bit)):
            if i in dims:
                spatial_shape.append(1)
                repeat_shape.append(shape_32bit[i])
            else:
                spatial_shape.append(shape_32bit[i])
                repeat_shape.append(1)
        task_layout = repeat(*repeat_shape) * spatial(*spatial_shape)

        grid_size = (remain_layout.num_workers + block_size - 1) // block_size
        accumulate_dtype = self.attributes['accumulate_dtype']
        reduce_type = self.attributes['reduce_type']
        ro = ReduceOperation.from_name(reduce_type)

        with hidet.script_module() as module:

            @hidet.script
            def reduce_kernel(x: f16[x.const_shape()], y: f16[y.const_shape()]):
                # Each 256-thread ThreadBlock handles 512 columns
                attr.cuda_grid_dim = grid_size
                attr.cuda_block_dim = block_size
                attr.cuda_min_blocks = 1

                x_f32 = tensor_pointer('float32', shape=shape_32bit)
                y_f32 = tensor_pointer('float32', shape=remain_shape_32bit)
                x_f32 = x
                y_f32 = y

                reg32 = register_tensor(f32, [1])
                regs16 = tensor_pointer('float16', shape=[2])
                regs16 = reg32
                rv = register_tensor(accumulate_dtype, [2])
                rv[0] = ro.initial_value(data_type(accumulate_dtype))
                rv[1] = ro.initial_value(data_type(accumulate_dtype))

                if threadIdx.x + blockIdx.x * block_size < remain_extent:
                    for indices in task_layout.on(threadIdx.x + blockIdx.x * block_size):
                        reg32[0] = x_f32.read(indices, protected=False)
                        rv[0] = ro.combine(rv[0], regs16[0])
                        rv[1] = ro.combine(rv[1], regs16[1])
                    regs16[0] = ro.finalize(acc=rv[0], size=reduce_extent)
                    regs16[1] = ro.finalize(acc=rv[1], size=reduce_extent)
                    for indices in remain_layout.on(threadIdx.x + blockIdx.x * block_size):
                        y_f32.write(indices, reg32[0], protected=False)

        ir_module = module.ir_module()
        add_packed_func(ir_module, func=reduce_kernel, pack_func_name=self.name)
        return ir_module


class ReduceBaseF16Op(Operator):
    def __init__(self, x: Tensor, dims: Optional[Sequence[int]], keep_dim: bool, reduce_type: ReduceType):
        rank = len(x.shape)
        if dims is None:
            dims = list(range(rank))
        dims = normalize_dim(dims, rank=rank)
        super().__init__(
            inputs=[x],
            task=ReduceF16Task(input_like(x, 'x'), dims, keep_dim, reduce_type),
            attributes={'dims': dims, 'keepdims': keep_dim},
        )


class ReduceMeanF16Op(ReduceBaseF16Op):
    def __init__(self, x: Tensor, dims: Optional[Sequence[int]], keepdims: bool = False):
        super().__init__(x, dims, keepdims, ReduceType.Average)


class ReduceSumF16Op(ReduceBaseF16Op):
    def __init__(self, x: Tensor, dims: Optional[Sequence[int]], keepdims: bool = False):
        super().__init__(x, dims, keepdims, ReduceType.Sum)


class ReduceMaxF16Op(ReduceBaseF16Op):
    def __init__(self, x: Tensor, dims: Optional[Sequence[int]], keepdims: bool = False):
        super().__init__(x, dims, keepdims, ReduceType.Max)


class ReduceMinF16Op(ReduceBaseF16Op):
    def __init__(self, x: Tensor, dims: Optional[Sequence[int]], keepdims: bool = False):
        super().__init__(x, dims, keepdims, ReduceType.Min)


class ReduceOrF16Op(ReduceBaseF16Op):
    def __init__(self, x: Tensor, dims: Optional[Sequence[int]], keepdims: bool = False):
        super().__init__(x, dims, keepdims, ReduceType.Or)


class ReduceAndF16Op(ReduceBaseF16Op):
    def __init__(self, x: Tensor, dims: Optional[Sequence[int]], keepdims: bool = False):
        super().__init__(x, dims, keepdims, ReduceType.And)


class ReduceProdF16Op(ReduceBaseF16Op):
    def __init__(self, x: Tensor, dims: Optional[Sequence[int]], keepdims: bool = False):
        super().__init__(x, dims, keepdims, ReduceType.Product)


def reduce_f16(x: Tensor, dims: Union[int, Sequence[int]], keepdims: bool, reduce_type: ReduceType) -> Tensor:
    if x.dtype != dtypes.float16:
        raise ValueError('reduce_f16 only support float16, got {}'.format(x.dtype))
    if x.shape[-1] % 2 != 0:
        raise ValueError('Expect the last dimension of the input tensors to be a multiple of 2')
    if isinstance(dims, int):
        dims = [dims]
    op_dict = {
        ReduceType.Sum: ReduceSumF16Op,
        ReduceType.Average: ReduceMeanF16Op,
        ReduceType.Max: ReduceMaxF16Op,
        ReduceType.Min: ReduceMinF16Op,
        ReduceType.Product: ReduceProdF16Op,
        ReduceType.Or: ReduceOrF16Op,
        ReduceType.And: ReduceAndF16Op,
    }
    op = op_dict[reduce_type]
    return op(x, dims, keepdims).get_output(0)
