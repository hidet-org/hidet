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
from typing import Union, List
from hidet import ir
from hidet.ir.type import DataType, int32
from hidet.ir.expr import cast, if_then_else
from hidet.ir.compute.primitives import TensorNode, compute
from hidet.ir import primitives as prim
from hidet.ir.compute import cops

from hidet.graph.ops.utils import Task, Operator, Tensor, input_like, normalize_dim


class SymmetricQuantizationTask(Task):
    def __init__(self, w: TensorNode, quant_type: DataType, dims: Union[int, List[int]] = -1):
        dims = normalize_dim(dims, len(w.shape))
        if not isinstance(dims, (list, tuple)):
            dims = [dims]

        wm = compute(
            name='abs', shape=w.shape, fcompute=lambda *indices: if_then_else(w[indices] >= 0, w[indices], -w[indices])
        )
        scale = cops.reduce(wm, dims, keep_dim=False, reduce_type='max')
        scale = compute(
            name='scaling', shape=scale.shape, fcompute=lambda *indices: scale[indices] / quant_type.max_value
        )

        def scale_weight(*indices):
            scale_indices = [indices[i] for i in range(len(indices)) if not i in dims]
            # Have to cast to int32 first because there are several ways convert bf16 to int8
            cast_to_int = cast(prim.round(w[indices] / scale[scale_indices]), int32)
            return cast(cast_to_int, quant_type)

        wq = compute(name='quantize', shape=w.shape, fcompute=scale_weight)
        super().__init__(
            name='symmetric_quantization',
            inputs=[w],
            outputs=[wq, scale],
            attributes={'dims': dims, 'quant_type': quant_type},
        )


class SymmetricDeQuantizationTask(Task):
    def __init__(self, wq: TensorNode, scale: TensorNode, dims: Union[int, List[int]] = -1):
        dims = normalize_dim(dims, len(wq.shape))
        if not isinstance(dims, (list, tuple)):
            dims = [dims]

        def unscale_weight(*indices):
            scale_indices = [indices[i] for i in range(len(indices)) if not i in dims]
            return cast(wq[indices], scale.type.dtype) * scale[scale_indices]

        w = compute(name='dequantize', shape=wq.shape, fcompute=unscale_weight)
        super().__init__(name='symmetric_dequantization', inputs=[wq, scale], outputs=[w], attributes={'dims': dims})


class SymmetricQuantizationOp(Operator):
    def __init__(self, w: Tensor, quant_type: ir.type.DataType, dims: Union[int, List[int]] = -1):
        super().__init__(
            inputs=[w],
            attributes={'dims': dims, 'quant_type': quant_type},
            task=SymmetricQuantizationTask(input_like(w, 'w'), quant_type=quant_type, dims=dims),
        )


class SymmetricDeQuantizationOp(Operator):
    def __init__(self, wq: Tensor, scale: Tensor, dims: Union[int, List[int]] = -1):
        super().__init__(
            inputs=[wq, scale],
            attributes={'dims': dims},
            task=SymmetricDeQuantizationTask(input_like(wq, 'wq'), input_like(scale, 'scale'), dims=dims),
        )


def symmetric_quantize(w: Tensor, quant_type: Union[str, ir.type.DataType] = 'int8', dims: Union[int, List[int]] = -1):
    op = SymmetricQuantizationOp(w, ir.type.data_type(quant_type), dims=dims)
    return op.outputs[0], op.outputs[1]


def symmetric_dequantize(wq: Tensor, scale: Tensor, dims: Union[int, List[int]] = -1):
    return SymmetricDeQuantizationOp(wq, scale, dims).outputs[0]
