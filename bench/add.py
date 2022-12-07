import hidet
import numpy as np
from typing import List, Callable, Any, Union, Optional, Dict

from hidet.ir import IRModule
from hidet.ir.builders import FunctionBuilder
from hidet.ir.expr import scalar_var, convert, Expr, And, cast
from hidet.ir.mapping import TaskMapping
from hidet.ir.primitives import block_idx, thread_idx
from hidet.ir.compute import ReduceOperation
from hidet.ir.stmt import AssignStmt, BufferStoreStmt, DeclareStmt
from hidet.ir.type import data_type
from hidet.ir.utils import index_deserialize
from hidet.graph.ops.definitions.arithmetic import BinaryElementwiseTask
from hidet.graph.ops.schedules.common import params_from_task
from hidet.utils import prod
from hidet.transforms.tools import fuse_and_pack

from hidet.lang import f16, f32, spatial, repeat, tensor, attr, grid, printf, cast
from hidet.lang.mapping import repeat, spatial
from hidet.lang.cuda import blockIdx, threadIdx, syncthreads
from hidet.lang.cuda import MmaConfig, mma_sync
from hidet.transforms.tools import add_packed_func


from hidet.ir import primitives
from hidet.ir import expr
from hidet.ir.expr import const_like, if_then_else
from hidet.utils import prod
from hidet.graph.tensor import convert
from hidet.graph.ops.definitions.utils import Task, Operator, Tensor, TensorNode, InverseMap, compute, input_like, broadcast_shape, broadcast_shapes, broadcast_indices


class AddTask(Task):
    def __init__(self, name: str, x: TensorNode, y: TensorNode, op: Callable[[Any, Any], Any]):
        x_shape = x.const_shape()
        y_shape = y.const_shape()
        z_shape = broadcast_shape(x_shape, y_shape)

        z = compute(
            name='z',
            shape=z_shape,
            fcompute=lambda *indices: op(
                x[broadcast_indices(indices, x_shape, z_shape)], y[broadcast_indices(
                    indices, y_shape, z_shape)]
            ),
        )

        super().__init__(
            name=name,
            inputs=[x, y],
            outputs=[z],
            inverse_map={
                v: InverseMap.identity(len(v_shape))
                for v, v_shape in zip([x, y], [x_shape, y_shape])
                if prod(v_shape) == prod(z_shape)
            },
        )

    def implement_cuda(self, workding_dir: str) -> IRModule:
        return cuda_schedule_add(self)


def cuda_schedule_add(task: AddTask) -> IRModule:
    x, y = task.inputs[0], task.inputs[1]
    z = task.outputs[0]
    x_dtype = x.ttype.dtype
    shape: List[int] = x.const_shape()
    total_elems = prod(shape)
    block_dim = 512
    num_dims = len(shape)
    mapping = spatial(*shape)
    with hidet.script_module() as module:
        @hidet.script
        def add_kernel(
            a: f32[tuple(shape)],
            b: f32[tuple(shape)],
            c: f32[tuple(shape)]
        ):
            attr.cuda_grid_dim = (total_elems + block_dim - 1) / block_dim
            attr.cuda_block_dim = block_dim
            # regs_c = tensor('shared', 'float32', shape)
            if blockIdx.x * block_dim + threadIdx.x < total_elems:
                for i, j, k in mapping.on(blockIdx.x * block_dim + threadIdx.x):
                    c.write([i, j, k], a[i, j, k] + b[i, j, k])
            pass

    ir_module = module.ir_module()
    add_packed_func(ir_module, func=add_kernel, pack_func_name=task.name)
    return ir_module


class AddOp(Operator):
    def __init__(self, x: Tensor, y: Tensor):
        super().__init__(inputs=[x, y], task=AddTask('add', input_like(
            x, 'x'), input_like(y, 'y'), op=lambda a, b: a + b))


hidet.option.search_space(2)
hidet.option.save_lower_ir(True)
hidet.option.cache_dir('.')

a = hidet.randn([128, 256, 512], dtype='float32')
b = hidet.randn([128, 256, 512], dtype='float32')
r = hidet.randn([2, 4, 6, 8, 10], dtype='float32')

print(AddOp(a, b).latency())
c = AddOp(a, b).get_output(0)
numpy_c = np.add(a.numpy(), b.numpy())
np.testing.assert_allclose(actual=c.cpu().numpy(),
                           desired=numpy_c, atol=1e-5, rtol=1e-5)

# c = hidet.ops.reduce_sum(r,[1,4])
# print(c)
