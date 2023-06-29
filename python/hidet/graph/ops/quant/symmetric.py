from typing import Any, Dict, Optional, Sequence, Union
from hidet import ir
from hidet.ir.expr import Expr, Var, is_constant, cast
from hidet.ir.compute.primitives import TensorNode, compute, reduce
from hidet.ir.utils import broadcast_shape, broadcast_indices
from hidet.ir import primitives as prim
from hidet.ir.compute import cops

from hidet.graph.ops.utils import Task, Operator, Tensor, input_like
from hidet.graph.ops.utils import TensorInput


class SymmetricQuantizationTask(Task):
    def __init__(self, w: TensorNode, quant_type: ir.type.DataType, dims=[-1]):
        dims = [i if i >= 0 else len(w.shape) + i for i in dims]
        self._assert(all(i >= 0 or i < len(w.shape) for i in dims), "dims are out of bounds")
        
        wm = compute(name='max', shape=w.shape, fcompute=lambda *indices: prim.abs(w[indices]))
        scale = cops.reduce_cop(wm, dims, keep_dim=False, reduce_type='max')
        scale = compute(name='scaling', shape=scale.shape, fcompute=lambda *indices: quant_type.max_value / scale[indices])

        def scale_weight(*indices):
            scale_indices = [indices[i] for i in range(len(indices)) if not i in dims]
            return cast(prim.round(w[indices] * scale[scale_indices]), quant_type)
        
        wq = compute(
            name='quantize',
            shape=w.shape,
            fcompute=scale_weight
        )
        super().__init__(
            name='symmetric_quantization_task',
            inputs=[w], outputs=[wq, scale],
            attributes={'dims': dims, 'quant_type': quant_type}
        )

class SymmetricDeQuantizationTask(Task):
    def __init__(self, wq: TensorNode, scale: TensorNode, dims=[-1]):
        dims = [i if i >= 0 else len(wq.shape) + i for i in dims]
        self._assert(all(i >= 0 or i < len(w.shape) for i in dims), "dims are out of bounds")

        def unscale_weight(*indices):
            scale_indices = [indices[i] for i in range(len(indices)) if not i in dims]
            return cast(wq[indices], scale.type.dtype) / scale[scale_indices]
        
        w = compute(
            name='dequantize',
            shape=wq.shape,
            fcompute=unscale_weight
        )
        super().__init__(
            name='symmetric_dequantization_task',
            inputs=[wq, scale], outputs=[w],
            attributes={'dims': dims}
        )


class SymmetricQuantizationOp(Operator):
    def __init__(self, w: Tensor, quant_type: ir.type.DataType, dims=[-1]):
        super().__init__(inputs=[w], 
                         attributes={'dims': dims, 'quant_type': quant_type}, 
                         task=SymmetricQuantizationTask(input_like(w, 'w'), quant_type=quant_type, dims=dims))


class SymmetricDeQuantizationOp(Operator):
    def __init__(self, wq: Tensor, scale: Tensor, dims=[-1]):
        super().__init__(inputs=[wq, scale], attributes={'dims': dims}, 
                         task=SymmetricDeQuantizationTask(input_like(wq, 'wq'), 
                                                          input_like(scale, 'scale'),
                                                          dims=dims))


def symmetric_quantize(w: Tensor, quant_type: Union[str, ir.type.DataType] = 'int8', dims=[-1]):
    op = SymmetricQuantizationOp(w, ir.type.data_type(quant_type), dims=dims)
    return op.get_output(0), op.get_output(1)


def symmetric_dequantize(wq: Tensor, scale: Tensor, dims=[-1]):
    return SymmetricDeQuantizationOp(wq, scale, dims).get_output(0)
