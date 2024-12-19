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
import hidet
from hidet.ir.expr import Expr
from hidet.ir.primitives.cuda.atomic import atomic_add
from hidet.lang import attrs
from hidet.lang.cuda import blockIdx, threadIdx
from hidet.ir.dtypes import int64
from hidet.utils import cdiv
from hidet.graph.ops.opaque import OpaqueOperator
from .utils import Tensor
from ...ir import IRModule
from ...ir.library import tune


def offset_helper(linear_idx: Expr, input_shape, input_stride):
    input_tensor_offset = 0

    dim_idx = len(input_shape) - 1
    while dim_idx >= 0:
        curr_dim_idx = linear_idx % input_shape[dim_idx]
        linear_idx = linear_idx // input_shape[dim_idx]
        input_tensor_offset += curr_dim_idx * input_stride[dim_idx]
        dim_idx -= 1

    return input_tensor_offset


class ScatterBaseOp(OpaqueOperator):
    def __init__(self, input: Tensor, index: Tensor, src: Tensor, dim: int, fname: str, inplace: bool = False):
        # fname can be one of: 'sum', 'prod', 'mean', 'amax', 'amin', 'replace'.
        # In the interest of time, only 'sum' is supported by Hidet currently.
        # fname = 'replace' can be used to support `torch.scatter`.

        if fname not in ['sum', 'replace']:
            raise ValueError(f"Scatter: fname {fname} is not supported by Hidet currently.")

        self.scatter_dim = dim
        self.fname = fname
        self.inplace = inplace

        share_map = {0: 0} if inplace else None

        super().__init__(
            name=f'scatter_{fname}_dim_{dim}{"_inplace" if inplace else ""}',
            inputs={'input': input, 'index': index, 'src': src},
            attributes={'dim': dim, 'fname': fname, 'inplace': inplace},
            share_map=share_map,
        )

    def symbolic_forward(self, input: Tensor, index: Tensor, src: Tensor):
        assert (
            index.shape == src.shape
        ), "scatter_add: For the time being, Hidet only supports index and src tensors of the same shape."

        return {'output': self.symbol(shape=input.shape, dtype=input.dtype, device=input.device)}

    def implement_cuda(self, inputs: List[Tensor], outputs: List[Tensor]) -> Union[IRModule, List[IRModule]]:
        return tune.extract_ir_modules(self.schedule_cuda)

    @tune.space(1)
    def schedule_cuda(self) -> IRModule:
        input_tensor = self.inputs[0]
        index_tensor = self.inputs[1]
        src_tensor = self.inputs[2]

        index_shape = index_tensor.shape
        index_numel = index_tensor.size

        src_shape, input_shape = src_tensor.shape, input_tensor.shape

        # calculate the "stride" along each dimension for index, src
        index_stride = [1]
        src_stride = [1]
        input_stride = [1]
        for i in range(len(index_shape) - 1, 0, -1):
            index_stride.insert(0, index_shape[i] * index_stride[0])
            src_stride.insert(0, src_shape[i] * src_stride[0])
            input_stride.insert(0, input_shape[i] * input_stride[0])

        # "re-shape" the input tensor for the purpose of properly calculating the offset
        input_shape = index_shape

        input_dim_stride = input_stride[self.scatter_dim]
        input_dim_size = input_tensor.shape[self.scatter_dim]

        index_size = input_dim_size
        index_stride = input_dim_stride

        # Set this stride to 0 as the index along this dimension
        # is determined by the corresponding element in the index tensor.
        input_stride[self.scatter_dim] = 0

        # same parameters as in PyTorch implementation
        nthread_per_block = 128
        work_per_thread = 4

        work_per_block = nthread_per_block * work_per_thread

        input_dtype = input_tensor.dtype
        src_dtype = src_tensor.dtype
        index_dtype = index_tensor.dtype

        num_blocks = cdiv(index_numel, nthread_per_block * work_per_thread)

        with hidet.script_module() as script_module:

            @hidet.script
            def scatter_internal_func(
                input: ~input_dtype,
                index: ~index_dtype,
                src: ~src_dtype,
                out: ~input_dtype,
                index_size: int64,
                total_elements: int64,
            ):
                attrs.func_kind = 'cuda_kernel'
                attrs.cuda.block_dim = nthread_per_block
                attrs.cuda.grid_dim = num_blocks

                linear_idx = work_per_block * blockIdx.x + threadIdx.x

                _ = input

                work_i = 0
                while work_i < work_per_thread:
                    if linear_idx < total_elements:

                        # this "linear idx" is the index into the
                        # "flattened" index tensor & src tensor.
                        # We need to re-calculate the offset for the input tensor
                        # using this index as well as the value of the index tensor at that index.

                        index_tensor_offset = linear_idx
                        input_tensor_offset = offset_helper(linear_idx, input_shape, input_stride)

                        index_on_scattered_dim = index[index_tensor_offset]

                        input_tensor_offset += index_on_scattered_dim * input_dim_stride

                        if self.fname == 'replace':
                            out[input_tensor_offset] = src[index_tensor_offset]
                        elif self.fname == 'sum':
                            atomic_add(~out[input_tensor_offset], src[index_tensor_offset])

                    linear_idx += nthread_per_block
                    work_i += 1

            @hidet.script
            def launch(input: ~input_dtype, index: ~index_dtype, src: ~src_dtype, out: ~input_dtype):
                attrs.func_kind = 'public'

                scatter_internal_func(input, index, src, out, index_size, index_numel)

        return script_module.ir_module()


def scatter_add_(input: Tensor, dim: int, index: Tensor, src: Tensor):
    return ScatterBaseOp(input, index, src, dim, 'sum', inplace=True).outputs[0]
