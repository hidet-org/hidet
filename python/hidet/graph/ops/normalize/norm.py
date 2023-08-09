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
from typing import List, Union
from hidet.ir import primitives as prim
from hidet.ir.library import tune
from hidet.ir.module import IRModule
from hidet.ir.primitives import active_mask, shfl_down_sync
from hidet.ir.compute import reduce
from hidet.ir.expr import Expr, convert, is_constant, if_then_else
from hidet.ir.type import DataType
from hidet.ir.layout import row_major
from hidet.ir.dtypes.vector import VectorType
from hidet.lang import spatial, repeat, grid, cast, register_tensor
from hidet.lang import data_type, TensorType, tensor_pointer, address, i32, attrs
from hidet.lang.cuda import blockIdx, threadIdx
from hidet.lang.cuda import dynamic_shared_memory, syncthreads
from hidet.graph.ops.utils import Task, Operator, Tensor, TensorNode
from hidet.graph.ops.utils import compute, input_like, normalize_dim
from hidet.utils import prod
from hidet.lang import float32


class NormalizeTask(Task):
    """
    Performs the following operation
        mean = x.mean(dims, keep_dim=True)
        x = x - mean
        variance = square(x).mean(dims, keep_dim=True)
        x = x * rsqrt(variance + epsilon)
    """

    def __init__(self, x: TensorNode, dims: List[int], epsilon: float, accumulate_dtype: str):
        dtype_str = x.type.dtype.name
        x_shape = x.shape
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
        return True

    def implement_cuda(self, working_dir: str):
        return tune.extract_ir_modules(self.norm_by_warp)

    @tune.space(2, two_shuffle=[True, False])
    @tune.space(1, two_shuffle=[True, False])
    def norm_by_warp(self, two_shuffle=False) -> IRModule:
        import hidet

        x, y = self.inputs[0], self.outputs[0]
        dtype = x.type.dtype

        lanes = 1
        input_shape: List[Expr] = list(x.shape)
        vtype: DataType = dtype
        if dtype.nbytes < 4:
            num_eles: int = 4 // dtype.nbytes
            if is_constant(input_shape[-1]) and input_shape[-1] % num_eles == 0:
                lanes = num_eles
                vtype = VectorType(dtype, lanes)

        read_shape = input_shape[:]
        read_shape[-1] /= lanes

        dims = self.dims

        spatial_shape = [v for i, v in enumerate(input_shape) if i not in dims]
        grid_size = prod(spatial_shape)

        warp_size = 32
        block_size = (read_shape[-1] + warp_size - 1) // warp_size * warp_size
        block_size = if_then_else(block_size > 512, 512, block_size)

        def get_mapping(tensor_shape):
            spatial_args = []
            repeat_args = []
            repeats = 1
            for idx, size in enumerate(tensor_shape):
                if idx not in dims:
                    repeat_args.append(1)
                    spatial_args.append(size)
                elif idx + 1 == len(input_shape):
                    repeats_per_block = (input_shape[-1] + block_size - 1) // block_size
                    repeat_args.append(repeats_per_block)
                    spatial_args.append(block_size)
                    repeats *= repeats_per_block
                else:
                    repeat_args.append(size)
                    spatial_args.append(1)
                    repeats *= size
            return repeat(*repeat_args) * spatial(*spatial_args), repeats

        read_mapping, read_block_repeat = get_mapping(read_shape)
        read_layout = row_major(*read_shape)

        # we choose to not vectorize the write so that epilogue can be fused
        write_mapping = read_mapping * repeat(*(1 for _ in range(len(read_shape) - 1)), lanes)
        write_layout = row_major(*input_shape)

        accumulate_dtype = data_type(self.attrs['accumulate_dtype'])
        smem_per_var = (block_size // warp_size) * accumulate_dtype.nbytes
        used_smem_bytes_per_block = 3 * smem_per_var

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
            def norm_kernel(x: dtype[x.shape], y: dtype[y.shape]):
                attrs.cuda.grid_dim = grid_size
                attrs.cuda.block_dim = block_size
                attrs.cuda.min_blocks = 1
                attrs.cuda.dynamic_smem_bytes = used_smem_bytes_per_block

                # this is used for staging warp shuffle results
                smem_mean = dynamic_shared_memory(byte_offset=0, dtype=accumulate_dtype)
                smem_m2 = dynamic_shared_memory(byte_offset=smem_per_var, dtype=accumulate_dtype)
                smem_count = dynamic_shared_memory(byte_offset=smem_per_var * 2, dtype=i32)

                x_vectorized = tensor_pointer(vtype, shape=read_shape, init=cast(x, ~vtype))
                # this is needed because normalization needs the original inputs
                read_cache = register_tensor(dtype, shape=[read_block_repeat * lanes])

                mean = register_tensor(accumulate_dtype, [1])
                m2 = register_tensor(accumulate_dtype, [1])
                count = register_tensor('int32', [1])

                other_mean = register_tensor(accumulate_dtype, [1])
                other_m2 = register_tensor(accumulate_dtype, [1])
                other_count = register_tensor('int32', [1])

                mean[0] = accumulate_dtype.zero
                m2[0] = accumulate_dtype.zero
                count[0] = 0
                k_read = 0
                for indices in read_mapping.on(threadIdx.x + blockIdx.x * block_size):
                    # read vectorized
                    if read_layout.within_bound(indices):
                        vec_read = x_vectorized[indices]
                        # local reduction of vectorized
                        if lanes > 1:
                            for lane_id in grid(lanes, "u+"):
                                lane_val = cast(address(vec_read), ~vtype.lane_type)[lane_id]
                                other_mean[0] = lane_val
                                other_m2[0] = accumulate_dtype.zero
                                other_count[0] = 1
                                welford_combine(mean, m2, count, other_mean, other_m2, other_count)
                                read_cache[k_read] = lane_val
                                k_read += 1
                        else:
                            other_mean[0] = vec_read
                            other_m2[0] = accumulate_dtype.zero
                            other_count[0] = 1
                            welford_combine(mean, m2, count, other_mean, other_m2, other_count)
                            read_cache[k_read] = vec_read
                            k_read += 1

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

                if threadIdx.x % warp_size == 0:
                    smem_mean[threadIdx.x // warp_size] = mean[0]
                    smem_m2[threadIdx.x // warp_size] = m2[0]
                    smem_count[threadIdx.x // warp_size] = count[0]
                syncthreads()

                if two_shuffle:
                    if threadIdx.x < warp_size:
                        if threadIdx.x < block_size // warp_size:
                            mean[0] = smem_mean[threadIdx.x]
                            m2[0] = smem_m2[threadIdx.x]
                            count[0] = smem_count[threadIdx.x]
                        else:
                            mean[0] = accumulate_dtype.zero
                            m2[0] = accumulate_dtype.zero
                            count[0] = 0

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
                else:
                    if threadIdx.x == 0:
                        for idx in range(block_size // warp_size):
                            if idx > 0:
                                other_mean[0] = smem_mean[idx]
                                other_m2[0] = smem_m2[idx]
                                other_count[0] = smem_count[idx]
                                welford_combine(smem_mean, smem_m2, smem_count, other_mean, other_m2, other_count)
                syncthreads()
                mean_final = smem_mean[0]
                m2_final = smem_m2[0]
                count_final = smem_count[0]
                # end of mean and var calculation, perform write back
                m2_final = m2_final / cast(count_final, accumulate_dtype)

                k_write = 0
                for write_idx in write_mapping.on(threadIdx.x + blockIdx.x * block_size):
                    if write_layout.within_bound(write_idx):
                        cached = read_cache[k_write]
                        normed = (cached - mean_final) * prim.rsqrt(
                            m2_final + cast(self.attrs['epsilon'], accumulate_dtype)
                        )
                        y.write(write_idx, normed)
                        k_write += 1

        ir_module = module.ir_module()
        return ir_module

    def implement_cpu(self, working_dir: str) -> Union[IRModule, List[IRModule]]:
        if self.dims[-1] != len(self.inputs[0].shape) - 1 or self.inputs[0].type.dtype != float32:
            return NotImplemented
        return tune.extract_ir_modules(self.schedule_norm_cpu)

    @tune.space(2, nthreads=['', 4, 8, 16, 32, 64, 96])
    @tune.space(1, nthreads=['', 8, 16])
    def schedule_norm_cpu(self, nthreads='') -> IRModule:
        import hidet
        from hidet.ir.primitives.cpu.avx import avx_f32x8_subtract, avx_f32x8_load, avx_f32x8_setzero, avx_f32x8_store,\
            avx_f32x8_add, avx_f32x8_set1, avx_f32x8_divide, avx_f32x8_multiply, avx_f32x8_find_sum, avx_f32x8_sqrt
        from hidet.ir.dtypes import float32
        from hidet.utils import prod

        shape = self.inputs[0].shape
        head = shape[:-len(self.dims)]
        head_size = prod(head)
        tail_size = prod(shape[-len(self.dims):])
        pre_tail = shape[-len(self.dims):-1]
        pre_tail_size = prod(pre_tail)
        with hidet.script_module() as module:

            @hidet.script
            def norm_cpu_kernel(x: float32[shape], out: float32[shape]):
                para = "p" + str(nthreads)
                for k in grid(head_size, attrs=para):
                    pre_tail_idx = spatial(*pre_tail).map(pre_tail_size) 
                    
                    offset = k * tail_size
                    head_idx = spatial(*head).map(k)
                    
                    mean_vec = avx_f32x8_setzero()
                    M2_vec = avx_f32x8_setzero()
                    epsilon_vec = avx_f32x8_set1(self.attrs['epsilon'])

                    mean_combined = 0.0
                    M2_combined = 0.0
                    if tail_size >= 8:
                        for i in range(tail_size // 8):
                            # welford algorithm
                            n_vec = avx_f32x8_set1(cast(i + 1, float32))
                            data_vec = avx_f32x8_load(x + offset + i * 8)
                            delta = avx_f32x8_subtract(data_vec, mean_vec)
                            mean_vec = avx_f32x8_add(mean_vec, avx_f32x8_divide(delta, n_vec))
                            delta2 = avx_f32x8_subtract(data_vec, mean_vec)
                            M2_vec = avx_f32x8_add(M2_vec, avx_f32x8_multiply(delta, delta2))

                        # welford combine
                        # TODO: case for numerical stability? (number too high for large matrix)
                        # TODO: look at the cascade thing in pytorch github
                        mean_combined = avx_f32x8_find_sum(mean_vec) / 8
                        mean_combined_vec = avx_f32x8_set1(mean_combined)
                        delta_vec = avx_f32x8_subtract(mean_vec, mean_combined_vec)
                        M2_combined = avx_f32x8_find_sum(M2_vec) + avx_f32x8_find_sum(avx_f32x8_multiply(delta_vec, delta_vec)) \
                            * (tail_size // 8)
                    mean_tail = 0.0
                    M2_tail = 0.0
                    # welford on remaining parts past 8
                    for i in range(tail_size % 8):
                        delta_tail = x[head_idx][pre_tail_idx][tail_size - tail_size % 8 + i] - mean_tail
                        mean_tail += delta_tail / cast(i+1, float32)
                        delta_tail2 = x[head_idx][pre_tail_idx][tail_size - tail_size % 8 + i] - mean_tail
                        M2_tail += delta_tail * delta_tail2
                    # welford combine vectorized and unvectorized
                    delta_end = mean_tail - mean_combined
                    mean = (mean_combined * (tail_size - tail_size % 8) + mean_tail * (tail_size % 8)) / tail_size
                    var = (M2_combined + M2_tail + delta_end * delta_end * (tail_size - tail_size % 8) * (tail_size % 8)
                           / tail_size) / tail_size
                    mean_vec = avx_f32x8_set1(mean)
                    var_vec = avx_f32x8_set1(var)
                    if tail_size >= 8:
                        for i in range(tail_size // 8):
                            # norm calculation
                            avx_f32x8_store(out + offset + i * 8,
                                            avx_f32x8_divide(avx_f32x8_subtract(avx_f32x8_load(
                                                x + offset + i * 8), mean_vec),
                                                avx_f32x8_sqrt(avx_f32x8_add(var_vec, epsilon_vec))))
                    for i in range(tail_size % 8):
                        out[head_idx][pre_tail_idx][tail_size - tail_size % 8 + i] =\
                            (x[head_idx][pre_tail_idx][tail_size - tail_size % 8 + i] - mean) *\
                            prim.rsqrt(var + self.attrs['epsilon'])

        norm_cpu_kernel.kind = "cpu_kernel"
        avx_f32x8_find_sum.kind = "cpu_internal"
        assert isinstance(norm_cpu_kernel, hidet.ir.Function)
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
    return NormalizeOp(x, axis, epsilon, accumulate_dtype).outputs[0]
