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
from typing import Optional, List, Union
import hidet
from hidet.ir.expr import cast
from hidet.lang import attrs
from hidet.lang.cuda import gridDim, blockIdx, blockDim, threadIdx
from hidet.ir.dtypes import float32, int64, int32
from hidet.utils import cdiv
from hidet.graph.ops.opaque import OpaqueOperator
from .utils import Tensor
from ...ir import IRModule
from ...ir.library import tune


class EmbeddingBagOp(OpaqueOperator):

    # TODO: For now, in the interest of time, I only supported the cases that are actually encountered while running
    # TODO: model from the TorchBench benchmark suite.
    # TODO: Need to go back to it later to support the operator fully.
    def __init__(self, input: Tensor, weight: Tensor, offsets: Tensor, mode: str):
        super().__init__(
            name=f'embedding_bag_{mode}',
            inputs={'input': input, 'weight': weight, 'offsets': offsets},
            attributes={'mode': mode},
        )

        assert input.dtype.is_integer(), "EmbeddingBag: expected `input` to be LongTensor"
        assert offsets.dtype.is_integer(), "EmbeddingBag: expected `offsets` to be LongTensor"
        assert weight.dtype.is_float(), "EmbeddingBag: expected `weight` to be FloatTensor"

        assert len(input.shape) == 1, "EmbeddingBag: Hidet only supports 1D input tensors for the time being."
        assert len(offsets.shape) == 1, "EmbeddingBag: `offsets` shape must be 1D."
        assert input.dtype.is_integer(), "EmbeddingBag: expected `input` to be LongTensor"
        assert offsets.dtype.is_integer(), "EmbeddingBag: expected `offsets` to be LongTensor"

        if mode == 'sum':
            self.mode = 0
        elif mode == 'mean':
            self.mode = 1
        else:
            assert False, "An error should have already be raised by the frontend"

    def symbolic_forward(self, input: Tensor, weight: Tensor, offsets: Tensor):
        num_bags = offsets.shape[0]
        embedding_dim = weight.shape[1]
        return {'output': self.symbol(shape=[num_bags, embedding_dim], dtype=weight.dtype, device=weight.device)}

    def implement_cuda(self, inputs: List[Tensor], outputs: List[Tensor]) -> Union[IRModule, List[IRModule]]:
        return tune.extract_ir_modules(self.schedule_cuda)

    @tune.space(1)
    def schedule_cuda(self) -> IRModule:

        indices, weight, offsets = self.inputs
        num_indices = indices.shape[0]
        nbags = offsets.shape[0]
        weight_nrows, feature_size = weight.shape
        reduction_mode = self.mode
        weight_stride0 = feature_size
        weight_stride1 = 1

        weight_dtype = weight.dtype
        index_dtype = indices.dtype
        offset_dtype = offsets.dtype

        with hidet.script_module() as script_module:

            @hidet.script
            def embedding_bag_kernel_sum_mean(
                inputs: ~index_dtype,
                weight: ~weight_dtype,
                offsets: ~offset_dtype,
                output: ~weight_dtype,
                num_indices: int64,
                num_bags: int64,
                feature_size: int64,
                weight_stride0: int64,
                weight_stride1: int64,
                mode: int32,
                num_rows: int64,
            ):
                attrs.func_kind = 'cuda_kernel'
                attrs.cuda.block_dim = 32, 8
                attrs.cuda.grid_dim = 1024

                chunks_per_bag = cdiv(feature_size, cast(blockDim.x, int64))
                num_chunks = num_bags * chunks_per_bag
                chunk_offset = blockIdx.x * blockDim.y + threadIdx.y
                chunk_stride = gridDim.x * blockDim.y

                MEAN_MODE = 1

                chunk = chunk_offset
                while chunk < num_chunks:
                    feature_dim = (chunk % chunks_per_bag) * blockDim.x + threadIdx.x
                    if feature_dim < feature_size:
                        bag = chunk // chunks_per_bag
                        weight_feat = weight + feature_dim * weight_stride1
                        begin = 0
                        if bag > 0:
                            begin = offsets[bag]

                        assert begin >= 0

                        end = num_indices
                        if bag < num_bags - 1:
                            end = offsets[bag + 1]
                        assert end >= begin
                        weight_feat_sum = 0.0
                        bag_size_ = 0.0
                        emb_idx = begin
                        while emb_idx < end:
                            assert inputs[emb_idx] < num_rows
                            weight_row = inputs[emb_idx] * weight_stride0
                            weight_value = weight_feat[weight_row]
                            weight_feat_sum += cast(weight_value, float32)

                            bag_size_ += 1
                            emb_idx += 1

                        if mode == MEAN_MODE:
                            weight_feat_sum //= cast(bag_size_, float32)

                        output[bag * feature_size + feature_dim] = weight_feat_sum

                    chunk += chunk_stride

            @hidet.script
            def launch(input: ~index_dtype, weight: ~weight_dtype, offsets: ~offset_dtype, output: ~weight_dtype):
                attrs.func_kind = 'public'
                embedding_bag_kernel_sum_mean(
                    input,
                    weight,
                    offsets,
                    output,
                    num_indices,
                    nbags,
                    feature_size,
                    weight_stride0,
                    weight_stride1,
                    reduction_mode,
                    weight_nrows,
                )

        return script_module.ir_module()


def embedding_bag(indices: Tensor, weight: Tensor, offsets: Optional[Tensor] = None, mode: str = 'mean') -> Tensor:
    return EmbeddingBagOp(indices, weight, offsets, mode).outputs[0]
