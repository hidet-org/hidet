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

from hidet.ir.compute import cops
from hidet.lang import grid
from hidet.lang.cuda import blockIdx, threadIdx, register_tensor
from hidet.ir.type import DataType
from hidet.ir.dtypes.vector import VectorType, vectorize
from hidet.ir.library import tune
from hidet.utils.py import cdiv
from ..arithmetic import square, sqrt
from ..utils import Task, Operator, Tensor, TensorNode, IRModule, ReduceType
from ..utils import compute, input_like, normalize_dim, arg_reduce


class ReduceTask(Task):
    def __init__(
        self, x: TensorNode, dims: List[int], keep_dim: bool, reduce_type: str, accumulate_dtype: str = 'float32'
    ):

        y = cops.reduce(x, dims, keep_dim, reduce_type, accumulate_dtype)
        self.dims: List[int] = dims
        self.keep_dim: bool = keep_dim
        self.reduce_type: str = reduce_type

        super().__init__(
            name='reduce_{}'.format(reduce_type),
            inputs=[x],
            outputs=[y],
            attributes={
                'dims': dims,
                'keep_dim': keep_dim,
                'reduce_type': reduce_type,
                'accumulate_dtype': accumulate_dtype,
            },
        )

    def allow_epilogue(self) -> bool:
        rank = len(self.inputs[0].shape)
        if rank - 1 in self.dims:  # pylint: disable=simplifiable-if-statement
            # use self.cuda_schedule_reduce_by_warp
            return True
        else:
            # use self.cuda_schedule_reduce_by_default
            return False

    def allow_prologue(self) -> bool:
        return False

    def implement_cuda(self, working_dir: str) -> Union[IRModule, List[IRModule]]:
        rank = len(self.inputs[0].shape)
        if rank - 1 in self.dims:
            return tune.extract_ir_modules(self.cuda_schedule_reduce_by_warp)
        else:
            return tune.extract_ir_modules(self.cuda_schedule_reduce_by_default)

    @tune.space(2, use_atomic=[True, False])
    @tune.space(1, use_atomic=[True, False])
    def cuda_schedule_reduce_by_warp(self, use_atomic=True) -> IRModule:
        import hidet
        from hidet.ir.primitives import active_mask, shfl_down_sync, shfl_sync
        from hidet.ir.expr import is_constant
        from hidet.ir.compute import ReduceOperation
        from hidet.ir.type import data_type, Int
        from hidet.ir.layout import row_major
        from hidet.lang import spatial, repeat, attrs, cast, tensor_pointer, address
        from hidet.lang.cuda import dynamic_shared_memory, syncthreads

        x, y = self.inputs[0], self.outputs[0]
        warp_size = 32
        xdtype = x.type.dtype
        shape: List[Int] = list(x.shape)
        lanes = 1
        vtype: Union[DataType, VectorType] = xdtype
        if xdtype.nbytes < 4:
            num_eles: int = 4 // xdtype.nbytes
            if is_constant(shape[-1]) and shape[-1] % num_eles == 0:
                lanes = num_eles
                vtype = vectorize(xdtype, lanes)
        read_shape = shape[:]
        read_shape[-1] //= lanes
        block_size = (read_shape[-1] + warp_size - 1) // warp_size * warp_size
        block_size = hidet.ir.expr.if_then_else(block_size > 1024, 1024, block_size)

        dims = self.dims
        if self.keep_dim:
            remain_shape = [v if i not in dims else 1 for i, v in enumerate(shape)]
        else:
            remain_shape = [v for i, v in enumerate(shape) if i not in dims]

        reduce_extent = hidet.utils.prod(shape[i] for i in dims)
        write_layout = spatial(*remain_shape)
        read_layout = row_major(*read_shape)

        spatial_shape = []
        repeat_shape = []
        for i in range(len(read_shape)):
            if i == len(read_shape) - 1:
                spatial_shape.append(block_size)
                repeat_shape.append((read_shape[i] + block_size - 1) // block_size)
            elif i in dims:
                spatial_shape.append(1)
                repeat_shape.append(read_shape[i])
            else:
                spatial_shape.append(read_shape[i])
                repeat_shape.append(1)
        task_layout = repeat(*repeat_shape) * spatial(*spatial_shape)
        grid_size = write_layout.num_workers
        accumulate_dtype = data_type(self.attrs['accumulate_dtype'])
        reduce_type = self.attrs['reduce_type']
        ro = ReduceOperation.from_name(reduce_type)

        perform_atomic_reduce = use_atomic and ro.has_atomic(accumulate_dtype)

        if perform_atomic_reduce:
            smem_needed = accumulate_dtype.nbytes
        else:
            smem_needed = accumulate_dtype.nbytes * block_size // warp_size

        with hidet.script_module() as module:

            @hidet.script
            def reduce_kernel(x: xdtype[x.shape], y: xdtype[y.shape]):
                attrs.cuda.grid_dim = grid_size
                attrs.cuda.block_dim = block_size
                attrs.cuda.min_blocks = 1
                attrs.cuda.dynamic_smem_bytes = smem_needed

                smem_staging = dynamic_shared_memory(byte_offset=0, dtype=accumulate_dtype)
                rv = ro.initial_value(data_type(accumulate_dtype))
                x_vectorized = tensor_pointer(vtype, shape=read_shape, init=cast(x, ~vtype))

                # initialize staging shared memory
                if perform_atomic_reduce:
                    if threadIdx.x == 0:
                        smem_staging[0] = rv
                else:
                    if threadIdx.x % 32 == 0:
                        smem_staging[threadIdx.x // 32] = rv

                for indices in task_layout.on(threadIdx.x + blockIdx.x * block_size):
                    if read_layout.within_bound(indices):
                        vec_read = x_vectorized[indices]
                        # vectorized load
                        if lanes > 1:
                            for lane_id in grid(lanes, "u+"):
                                lane_val = cast(address(vec_read), ~vtype.lane_type)[lane_id]
                                rv = ro.combine(rv, cast(lane_val, accumulate_dtype))
                        else:
                            rv = ro.combine(rv, cast(vec_read, accumulate_dtype))

                # Warp reduce by shuffle down
                mask = active_mask()
                rv = ro.combine(rv, shfl_down_sync(mask, rv, 16, 32))
                rv = ro.combine(rv, shfl_down_sync(mask, rv, 8, 32))
                rv = ro.combine(rv, shfl_down_sync(mask, rv, 4, 32))
                rv = ro.combine(rv, shfl_down_sync(mask, rv, 2, 32))
                rv = ro.combine(rv, shfl_down_sync(mask, rv, 1, 32))
                rv = shfl_sync(mask, rv, 0, 32)

                # write to staging area
                if perform_atomic_reduce:
                    syncthreads()
                if threadIdx.x % 32 == 0:
                    if perform_atomic_reduce:
                        ro.atomic_combine(~smem_staging[0], cast(rv, accumulate_dtype))
                    else:
                        smem_staging[threadIdx.x // 32] = rv

                # collect results from all warps
                syncthreads()
                if threadIdx.x == 0:
                    if perform_atomic_reduce:
                        rv = smem_staging[0]
                    else:
                        for i in range(block_size // warp_size):
                            if i > 0:
                                rv = ro.combine(rv, smem_staging[i])
                    rv = ro.finalize(acc=rv, size=reduce_extent)
                    for indices in write_layout.on(blockIdx.x):
                        y.write(indices, rv)

        ir_module = module.ir_module()
        return ir_module

    @tune.space(2, max_block_size=[256, 512, 1024], use_atomic=[True, False])
    @tune.space(1, max_block_size=[256, 512, 1024], use_atomic=[True, False])
    def cuda_schedule_reduce_by_default(self, max_block_size=256, use_atomic=True) -> IRModule:
        import hidet
        from hidet.ir.compute import ReduceOperation
        from hidet.ir.type import data_type, Int, tensor_type
        from hidet.ir.expr import cast, address, is_constant
        from hidet.ir.layout import row_major, local_layout
        from hidet.lang import spatial, repeat, attrs, tensor_pointer
        from hidet.lang.cuda import dynamic_shared_memory, syncthreads

        WARP_SIZE = 32
        max_num_warps = max_block_size // WARP_SIZE

        x, y = self.inputs[0], self.outputs[0]
        xdtype = x.type.dtype
        shape: List[Int] = list(x.shape)

        accumulate_dtype = data_type(self.attrs['accumulate_dtype'])
        reduce_type = self.attrs['reduce_type']
        ro = ReduceOperation.from_name(reduce_type)
        perform_atomic_reduce = use_atomic and ro.has_atomic(accumulate_dtype)

        lanes = 1
        vtype: Union[VectorType, DataType] = xdtype
        if xdtype.nbytes < 4:
            num_eles: int = 4 // xdtype.nbytes
            if shape[-1] % num_eles == 0:
                lanes = num_eles
                vtype = VectorType(xdtype, lanes)

        read_shape = shape[:]
        read_shape[-1] //= lanes
        x_vectorized_shape = read_shape[:]
        read_layout = row_major(*read_shape)

        dims = self.dims
        read_remain_shape = [v for i, v in enumerate(shape) if i not in dims]
        read_remain_shape[-1] //= lanes
        if self.keep_dim:
            write_remain_shape = [v if i not in dims else 1 for i, v in enumerate(shape)]
        else:
            write_remain_shape = read_remain_shape[:]
        y_vectorized_shape = write_remain_shape[:]

        reduce_shape = [v for i, v in enumerate(shape) if i in dims]
        reduce_extent = hidet.utils.prod(reduce_shape)
        remain_extent = hidet.utils.prod(read_remain_shape)

        remain_threads = hidet.ir.expr.if_then_else(remain_extent > max_block_size, max_block_size, remain_extent)
        remain_warps = cdiv(remain_threads, WARP_SIZE)
        max_reduce_warps = max_num_warps // remain_warps
        reduce_warps = cdiv(reduce_extent, WARP_SIZE)
        reduce_warps = hidet.ir.expr.if_then_else(max_reduce_warps < reduce_warps, max_reduce_warps, reduce_warps)
        repeats_per_reduce = cdiv(reduce_extent, reduce_warps)
        num_warps = reduce_warps * remain_warps
        block_size = num_warps * WARP_SIZE
        grid_size = cdiv(remain_extent, remain_warps * WARP_SIZE)

        remain_reduce_mapping = spatial(grid_size, 1) * spatial(remain_warps * WARP_SIZE, reduce_warps) * repeat(1, repeats_per_reduce)
        reduce_mapping = spatial(*reduce_shape)
        remain_mapping = spatial(*read_remain_shape)
        import pdb
        pdb.set_trace()

        write_task_mapping = spatial(*write_remain_shape)

        smem_bound_check_shape = []
        smem_flatten_idx_shape = []
        for i in range(len(read_shape)):
            if i == dims[0]:
                smem_bound_check_shape.append(reduce_warps)
                smem_flatten_idx_shape.append(1)
            elif i in dims:
                smem_bound_check_shape.append(1)
                smem_flatten_idx_shape.append(1)
            else:
                smem_bound_check_shape.append(shape[i])
                smem_flatten_idx_shape.append(shape[i])

        read_task_mapping =  spatial(*spatial_shape) * repeat(*repeat_shape)
        
        if remain_threads < remain_warps * WARP_SIZE:
            pad = remain_warps * WARP_SIZE - remain_threads
            spatial_shape[-1] = spatial_shape[-1] + pad
        # Since threadblocks are distributed unevenly across multiple dimensions,
        # we will use a flattened 1D layout for shared memory for simplicity.
        smem_length = remain_warps * WARP_SIZE * lanes
        smem_layout = row_major(smem_length)
        smem_bound_check_layout = row_major(*smem_bound_check_shape)
        smem_flatten_idx_layout = row_major(*smem_flatten_idx_shape)
        smem_repeat_shape = [1] * (len(spatial_shape) - 1) + [lanes]
        smem_task_mapping = spatial(*spatial_shape) * repeat(*smem_repeat_shape)
        smem_type = tensor_type(accumulate_dtype, layout=smem_layout)
        smem_needed = smem_type.storage_bytes()

        def flatten_smem_idx(indices):
            x = indices[:]
            x[dims[0]] = 1
            return smem_flatten_idx_layout.serialize(x) % smem_length
        
        def unflatten(indices):
            # indices should only contain a 2D coordinate in the (remain, reduce) space
            assert len(indices) == 2
            remain_indices = remain_mapping.map(indices[0])
            reduce_indices = reduce_mapping.map(indices[1])
            unflattened_indices = merge(remain_indices, reduce_indices)

            return unflattened_indices


        with hidet.script_module() as module:

            @hidet.script
            def reduce_kernel(x: xdtype[x.shape], y: xdtype[y.shape]):
                # Each 256-thread ThreadBlock handles 512 columns
                attrs.cuda.grid_dim = grid_size
                attrs.cuda.block_dim = block_size
                attrs.cuda.min_blocks = 1
                attrs.cuda.dynamic_smem_bytes = smem_needed

                smem_base = dynamic_shared_memory(byte_offset=0, dtype=accumulate_dtype)
                smem_staging = tensor_pointer(accumulate_dtype, layout=smem_layout, init=cast(smem_base, ~accumulate_dtype))

                x_vectorized = tensor_pointer(vtype, shape=x_vectorized_shape, init=cast(x, ~vtype))
                y_vectorized = tensor_pointer(vtype, shape=y_vectorized_shape, init=cast(y, ~vtype))

                rv = register_tensor(accumulate_dtype, [lanes])
                for lane_id in grid(lanes, "u+"):
                    rv[lane_id] = ro.initial_value(accumulate_dtype)

                write_val = register_tensor(vtype, [1])

                # Init smem - TODO
                for indices in smem_task_mapping.on(threadIdx.x + blockIdx.x * block_size):
                    if smem_bound_check_layout.within_bound(indices):
                        flattened_idx = flatten_smem_idx(indices)
                        smem_staging[flattened_idx] = rv[0]

                for flat_indices in remain_reduce_mapping.on(threadIdx.x + blockIdx.x * block_size):
                    tensor_indices = unflatten(flat_indices)
                    # ------------------------------------------
                    if read_layout.within_bound(indices):
                        vec_read = x_vectorized[indices]
                        if lanes > 1:
                            for lane_id in grid(lanes, "u+"):
                                lane_val = cast(address(vec_read), ~vtype.lane_type)[lane_id]
                                rv[lane_id] = ro.combine(rv[lane_id], cast(lane_val, accumulate_dtype))
                        else:
                            rv[0] = ro.combine(rv[0], cast(vec_read, accumulate_dtype))

                # At this point, all threads contain their local reduction value in their register rv[]
                # Next, need to reduce those values into respective smem location

                syncthreads()
                if perform_atomic_reduce:
                    for indices in smem_task_mapping.on(threadIdx.x + blockIdx.x * block_size):
                        if smem_bound_check_layout.within_bound(indices):
                            flattened_idx = flatten_smem_idx(indices)
                            ro.atomic_combine(~smem_staging[flattened_idx], rv[indices[-1] % lanes])
                else:
                    # Reduce via multiround writebacks + syncthreads
                    for k in range(reduce_warps):
                        for indices in smem_task_mapping.on(threadIdx.x + blockIdx.x * block_size):
                            if smem_bound_check_layout.within_bound(indices):
                                flattened_idx = flatten_smem_idx(indices)
                                reduce_round = indices[dims[0]]
                                if reduce_round == k:
                                    smem_staging[flattened_idx] = ro.combine(smem_staging[flattened_idx], rv[indices[-1] % lanes])
                        syncthreads()
                
                syncthreads()
                # At this point, the shared memory contains the final reduction value.
                # Next, need to write back to global memory

                if threadIdx.x + blockIdx.x * block_size < remain_extent:
                    for indices in smem_task_mapping.on(threadIdx.x + blockIdx.x * block_size):
                        if smem_bound_check_layout.within_bound(indices):
                            flattened_idx = flatten_smem_idx(indices)
                            if lanes > 1:
                                lane_vec = cast(~write_val, ~vtype.lane_type)
                                lane_vec[indices[-1] % lanes] = ro.finalize(acc=smem_staging[flattened_idx], size=reduce_extent)
                            else:
                                write_val[0] = ro.finalize(acc=smem_staging[flattened_idx], size=reduce_extent)
                    for indices in write_task_mapping.on(threadIdx.x + blockIdx.x * block_size):
                        y_vectorized[indices] = write_val[0]

        ir_module = module.ir_module()
        return ir_module


class ArgReduceTask(Task):
    def __init__(self, x: TensorNode, dim: int, keep_dim: bool, reduce_type: str):
        y_shape = []
        for i, extent in enumerate(x.shape):
            if i == dim:
                if keep_dim:
                    y_shape.append(1)
            else:
                y_shape.append(extent)

        def fcompute(*indices):
            def reduce_fcompute(reduce_index):
                x_indices = indices[:dim] + (reduce_index,) + indices[dim + (1 if keep_dim else 0) :]
                return x[x_indices]

            return arg_reduce(
                extent=x.shape[dim], fcompute=reduce_fcompute, reduce_type=reduce_type, index_dtype='int64'
            )

        y = compute(name='y', shape=y_shape, fcompute=fcompute)
        super().__init__(
            name='arg{}'.format(reduce_type),
            inputs=[x],
            outputs=[y],
            attributes={'dim': dim, 'reduce_type': reduce_type},
        )


class ReduceBaseOp(Operator):
    def __init__(self, x: Tensor, dims: Optional[Sequence[int]], keep_dim: bool, reduce_type: str):
        rank = len(x.shape)
        if dims is None:
            dims = list(range(rank))
        dims = normalize_dim(dims, rank=rank)
        super().__init__(
            inputs=[x],
            attributes={'dims': dims, 'keepdims': keep_dim},
            task=ReduceTask(input_like(x, 'x'), dims, keep_dim, reduce_type),
        )


class ArgReduceBaseOp(Operator):
    def __init__(self, x: Tensor, dim: int, keep_dim: bool, reduce_type: str):
        if reduce_type not in [ReduceType.Min.value, ReduceType.Max.value]:
            raise NotImplementedError('Do not support arg reduce type: {}'.format(reduce_type))
        dim = normalize_dim(dim, rank=len(x.shape))
        super().__init__(
            inputs=[x],
            attributes={'dim': dim, 'keepdims': keep_dim},
            task=ArgReduceTask(input_like(x, 'x'), dim, keep_dim, reduce_type),
        )


class ReduceMeanOp(ReduceBaseOp):
    def __init__(self, x: Tensor, dims: Optional[Sequence[int]], keepdims: bool = False):
        super().__init__(x, dims, keepdims, ReduceType.Average.value)


class ReduceSumOp(ReduceBaseOp):
    def __init__(self, x: Tensor, dims: Optional[Sequence[int]], keepdims: bool = False):
        super().__init__(x, dims, keepdims, ReduceType.Sum.value)


class ReduceMaxOp(ReduceBaseOp):
    def __init__(self, x: Tensor, dims: Optional[Sequence[int]], keepdims: bool = False):
        super().__init__(x, dims, keepdims, ReduceType.Max.value)


class ReduceMinOp(ReduceBaseOp):
    def __init__(self, x: Tensor, dims: Optional[Sequence[int]], keepdims: bool = False):
        super().__init__(x, dims, keepdims, ReduceType.Min.value)


class ReduceOrOp(ReduceBaseOp):
    def __init__(self, x: Tensor, dims: Optional[Sequence[int]], keepdims: bool = False):
        super().__init__(x, dims, keepdims, ReduceType.Or.value)


class ReduceAndOp(ReduceBaseOp):
    def __init__(self, x: Tensor, dims: Optional[Sequence[int]], keepdims: bool = False):
        super().__init__(x, dims, keepdims, ReduceType.And.value)


class ReduceProdOp(ReduceBaseOp):
    def __init__(self, x: Tensor, dims: Optional[Sequence[int]], keepdims: bool = False):
        super().__init__(x, dims, keepdims, ReduceType.Product.value)


class ArgMinOp(ArgReduceBaseOp):
    def __init__(self, x: Tensor, dim: int, keepdims: bool):
        super().__init__(x, dim, keepdims, ReduceType.Min.value)


class ArgMaxOp(ArgReduceBaseOp):
    def __init__(self, x: Tensor, dim: int, keepdims: bool):
        super().__init__(x, dim, keepdims, ReduceType.Max.value)


def mean(x: Tensor, dims: Union[int, List[int]], keep_dim: bool = False) -> Tensor:
    if isinstance(dims, int):
        dims = [dims]
    return ReduceMeanOp(x, dims, keep_dim).outputs[0]


def sum(x: Tensor, dims: Union[int, List[int]], keep_dim: bool = False) -> Tensor:
    if isinstance(dims, int):
        dims = [dims]
    return ReduceSumOp(x, dims, keep_dim).outputs[0]


def max(x: Tensor, dims: Union[int, List[int]], keep_dim: bool = False) -> Tensor:
    if isinstance(dims, int):
        dims = [dims]
    return ReduceMaxOp(x, dims, keep_dim).outputs[0]


def min(x: Tensor, dims: Union[int, List[int]], keep_dim: bool = False) -> Tensor:
    if isinstance(dims, int):
        dims = [dims]
    return ReduceMinOp(x, dims, keep_dim).outputs[0]


def var(x: Tensor, dims: Union[int, List[int]], keep_dim: bool = False) -> Tensor:
    x = x - x.mean(dims=dims, keep_dim=True)
    return square(x).mean(dims=dims, keep_dim=keep_dim)


def std(x: Tensor, dims: Union[int, List[int]], keep_dim: bool = False) -> Tensor:
    return sqrt(var(x, dims=dims, keep_dim=keep_dim))


def prod(x: Tensor, dims: Union[int, List[int]], keep_dim: bool = False) -> Tensor:
    if isinstance(dims, int):
        dims = [dims]
    return ReduceProdOp(x, dims, keep_dim).outputs[0]


def argmin(x: Tensor, dim: int, keep_dim: bool = False) -> Tensor:
    return ArgMinOp(x, dim, keep_dim).outputs[0]


def argmax(x: Tensor, dim: int, keep_dim: bool = False) -> Tensor:
    return ArgMaxOp(x, dim, keep_dim).outputs[0]


def all(x: Tensor, /, *, axis=None, keepdims=False) -> Tensor:
    """
    Check if all of the elements on the given axis evaluates to True.

    Parameters
    ----------
    x: Tensor
        The input tensor.

    axis: int or Sequence[int], optional
        The axis or axes along which to perform the logical AND. None indicates to perform the reduction on the whole
        tensor. When an integer or a sequence of integers are given, they must be in range [-N, N), where N is the
        rank of the input tensor.

    keepdims: bool, default=False
        Whehter to keep the dimension.

    Returns
    -------
    ret: Tensor
        The result of logical AND reduction with bool data type.
    """
    x = x.astype('bool')
    return ReduceAndOp(x, dims=axis, keepdims=keepdims).outputs[0]


def any(x: Tensor, /, *, axis=None, keepdims=False) -> Tensor:
    """
    Check if any of the elements on the given axis evaluates to True.

    Parameters
    ----------
    x: Tensor
        The input tensor.

    axis: int or Sequence[int], optional
        The axis or axes along which to perform the logical OR. None indicates to perform the reduction on the whole
        tensor. When an integer or a sequence of integers are given, they must be in range [-N, N), where N is the
        rank of the input tensor.

    keepdims: bool, default=False
        Whehter to keep the dimension.

    Returns
    -------
    ret: Tensor
        The result of logical OR reduction with bool data type.
    """
    x = x.astype('bool')
    return ReduceOrOp(x, dims=axis, keepdims=keepdims).outputs[0]
