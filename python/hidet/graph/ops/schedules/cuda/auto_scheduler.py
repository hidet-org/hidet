from typing import List, Dict

from hidet.ir.builders import FunctionBuilder
from hidet.ir.compute import TensorNode, GridCompute
from hidet.ir.expr import Call, Expr, Var
from hidet.ir.functors import collect, rewrite, simplify_to_int
from hidet.ir.stmt import Stmt, BufferStoreStmt, EvaluateStmt
from hidet.utils import prod
from ..auto_scheduler import AutoScheduler, ComputeExprLower


class CudaAutoScheduler(AutoScheduler):
    def schedule_grid_compute(self, gc: GridCompute, node: TensorNode, node_map: Dict[TensorNode, Expr]) -> Stmt:
        # pylint: disable=unnecessary-comprehension, import-outside-toplevel
        from hidet.ir.primitives.cuda import threadIdx, blockIdx
        from hidet.ir.mapping import row_spatial, TaskMapping
        used_tensors: List[TensorNode] = collect(gc.value, TensorNode, stop_when_found=True)
        param_tensors: List[TensorNode] = used_tensors + [node]
        params: List[Var] = [Var(tensor.name, tensor.data_type) for tensor in param_tensors]

        block_dim = 500
        grid_dim: int = simplify_to_int((prod(gc.shape) + block_dim - 1) // block_dim)

        with FunctionBuilder(name=f'compute_{node.name}', kind='cuda_kernel',
                             grid_dim=grid_dim, block_dim=block_dim) as fb:
            # set function parameters
            fb.extend_params(params)

            # calculate task indices assigned to current worker
            worker = blockIdx.x * block_dim + threadIdx.x

            mapping: TaskMapping = row_spatial(*gc.shape)
            iter_names = [f'i{i}' for i in range(len(gc.shape))]
            with fb.if_then(worker < mapping.num_workers):
                with fb.for_mapping(iter_names, mapping, worker) as task_index:
                    out_param: Var = params[-1]
                    param_map: Dict[TensorNode, Expr] = {tensor_node: param_var
                                                         for tensor_node, param_var in zip(param_tensors, params)}
                    compute_lower = ComputeExprLower(gc.value, param_map=param_map)
                    stmts, value = compute_lower.lower()
                    rmap = {axis: axis_value for axis, axis_value in zip(gc.axes, task_index)}
                    stmts, value = [rewrite(stmt, rmap) for stmt in stmts], rewrite(value, rmap)
                    fb += stmts
                    fb += BufferStoreStmt(out_param, task_index, value)
        func = fb.get()
        func_var = self.add_function(func)
        return EvaluateStmt(Call(func_var, args=[node_map[param_tensor] for param_tensor in param_tensors]))
