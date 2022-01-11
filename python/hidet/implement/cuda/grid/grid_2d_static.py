from typing import Mapping, List, Dict, Union

from hidet.ir.node import Node
from hidet.ir.type import scalar_type, tensor_type, TensorType
from hidet.ir.expr import Expr, Var, IntVar, Call, And, convert, Constant, Axis, if_then_else, TensorElement
from hidet.ir.stmt import LetStmt, IfStmt, EvaluateStmt, concat_stmts, SeqStmt, BufferStoreStmt
from hidet.ir.task import Task, Grid, ThreadBlock, Thread
from hidet.ir.func import IRModule, Function, FunctionGroup
from hidet.ir.dialects.compute import TensorInput, ScalarInput, TensorCompute, compute, scalar_input
from hidet.ir.dialects.pattern import TaskPattern, TensorComputePattern, ScalarExprPattern
from hidet.ir.dialects.lowlevel import VoidType, Address
from hidet.ir.functors import rewrite, infer_type, collect, simplify, coefficients
from hidet.implement.implementer import Implementer, implement, register_impl


class NotSupported(Exception):
    pass


@register_impl('cuda_grid_split_implementer')
class CudaGridSplitImplementer(Implementer):
    def __init__(self):
        self.n = Constant(None, dtype=scalar_type('int32'))
        self.m = Constant(None, dtype=scalar_type('int32'))
        self.axes = [Axis(self.n), Axis(self.m)]
        self.value = ScalarExprPattern()
        self.pattern = TaskPattern(
            compute_pattern=TensorCompute('out',
                                          shape=[self.n, self.m],
                                          axes=self.axes,
                                          value=self.value),
            required_params=[],
            required_param_types=[],
            allow_tensor_extra_params=True,
            worker=Grid(grid_dim=None, block_dim=None)
        )
        self.split_factors = [32, 48, 64, 96, 128, 192, 256]
        self.block_dims = [64, 128, 256, 512]

    def priority(self) -> int:
        return -1  # not finished

    def task_pattern(self) -> TaskPattern:
        return self.pattern

    def get_subtask(self,
                    subtask_name,
                    task: Task,
                    match: Mapping[Node, Node],
                    factor: int,
                    sub_shape: List[int],
                    block_dim: int,
                    n_block_idx: Var,
                    m_block_idx: Var,
                    func_params: List[Var]
                    ) -> Task:
        param2type = {param: param_type for param, param_type in zip(task.params, task.params_type)}
        param2var = {param: arg for param, arg in zip(task.params, func_params)}
        axes: List[Axis] = [match[axis] for axis in self.axes]
        value: Expr = match[self.value]
        subtask_axes = [Axis(sub_shape[0]), Axis(sub_shape[1])]
        try:
            # first try to avoid passing block index to subtask
            outer_axes = [n_block_idx, m_block_idx]
            tensor_inputs: List[TensorInput] = collect(value, TensorInput)
            tensor_elements: List[TensorElement] = collect(value, TensorElement)
            subtask_param2arg = {a: b for a, b in param2var.items()}
            rmap = {}
            for ti in tensor_inputs:
                tes = [te for te in tensor_elements if te.base is ti]
                if len(tes) == 0:
                    raise NotSupported("do not consider not-used tensor input")
                if len(tes) > 1:
                    raise NotSupported("do not support multiple access to the same tensor input as far as now")
                te = tes[0]
                indices = te.indices
                arg_indices = []
                for idx_expr in indices:
                    coeffs = coefficients(idx_expr, axes)
                    if any(sum(order) > 1 for order in coeffs):
                        raise NotSupported("do not support high order indexing e.g., A[i * i][j], A[i * j][k]")
                    for order, coeff in coeffs.items():
                        if sum(order) > 0 and any(v not in task.params for v in collect(coeff, [Var, ScalarInput])):
                            raise NotSupported("The coefficient is dependent on inner var, e.g., A[i * reduce_axis], where reduce_axis is a inner var")
                    s = convert(0)
                    for order in coeffs:
                        for i in range(2):
                            if order[i] == 1:
                                s = s + coeffs[order] * outer_axes[i] * factor
                    arg_indices.append(s)
                rmap[te] = rewrite(te, {a: b for a, b in zip(axes, subtask_axes)})
                subtask_param2arg[ti] = Address(TensorElement(param2var[ti], arg_indices))  # update
            subtask_value = rewrite(value, rmap)
            remaining_axes = collect(subtask_value, Axis)
            if any(axis in axes for axis in remaining_axes):
                raise NotSupported('There are axes in computation besides indexing. e.g., A[i] * i')

            subtask_compute = TensorCompute('out', sub_shape, subtask_axes, subtask_value)
            input_params = collect(subtask_value, (ScalarInput, TensorInput))
            subtask_params = input_params + [subtask_compute]
            param2type[subtask_compute] = param2type[task.compute]
            subtask_params_type = [param2type[param] for param in subtask_params]

            # out_arg = param2var[task.compute][n_block_idx: n_block_idx * factor + sub_shape[0], m_block_idx: m_block_idx + sub_shape[1]]
            out_arg = Address(TensorElement(param2var[task.compute], [n_block_idx * factor, m_block_idx * factor]))
            args = [subtask_param2arg[param] for param in input_params] + [out_arg]

            return Task(subtask_name, subtask_compute, subtask_params, subtask_params_type, ThreadBlock(block_dim)), args

        except NotSupported:
            # fallback subtask with block index
            outer_axes_params = [scalar_input('i_o', 'int32'), scalar_input('j_o', 'int32')]
            sub_compute = compute('out', sub_shape, lambda i, j: rewrite(value, {old_axis: inner_axis + outer_axis * factor for old_axis, inner_axis, outer_axis in zip(axes, [i, j], outer_axes_params)}))
            output_param = sub_compute

            other_params = collect(value, (ScalarInput, TensorInput))
            params = outer_axes_params + other_params + [output_param]
            task_ret_type: TensorType = param2type[task.compute]
            subtask_ret_type: TensorType = tensor_type(task_ret_type.scope, task_ret_type.scalar_type, [factor, factor], task_ret_type.strides)
            params_type = [scalar_type('int32'), scalar_type('int32')] + [param2type[param] for param in other_params] + [subtask_ret_type]

            # out_arg = param2var[task.compute][n_block_idx: n_block_idx + sub_shape[0], m_block_idx: m_block_idx + sub_shape[1]]
            out_arg = Address(TensorElement(param2var[task.compute], [n_block_idx * factor, m_block_idx * factor]))
            args = [n_block_idx, m_block_idx] + [param2var[param] for param in other_params] + [out_arg]

            return Task(subtask_name, sub_compute, params, params_type, ThreadBlock(block_dim)), args

    def implement(self, task: Task, match: Mapping[Node, Node]) -> IRModule:
        assert isinstance(task.compute, TensorCompute)
        ir_module = IRModule()
        func_name = f'{task.name}.grid'
        func_group = FunctionGroup(func_name)
        for factor in self.split_factors:
            for block_dim in self.block_dims:
                n: Constant = match[self.n]
                m: Constant = match[self.m]
                blockIdx = IntVar('blockIdx.x')

                func_params: List[Var] = [Var(param.name, param_type) for param, param_type in zip(task.params, task.params_type)]
                func_body_stmts = []
                func_local_vars = []
                n_grid_dim = (n.value + factor - 1) // factor
                m_grid_dim = (m.value + factor - 1) // factor
                grid_dim = n_grid_dim * m_grid_dim

                n_block_idx = IntVar('n_block_idx')
                m_block_idx = IntVar('m_block_idx')
                func_body_stmts.append(LetStmt(n_block_idx, blockIdx / m_grid_dim))
                func_body_stmts.append(LetStmt(m_block_idx, blockIdx % m_grid_dim))

                for i in range(2):
                    i_cond = [n.value >= factor, n.value % factor > 0]
                    i_shape = [factor, n.value % factor]
                    i_cond_expr = [(n_block_idx + 1) * factor <= n, n_block_idx + 1 == n_grid_dim]
                    for j in range(2):
                        j_cond = [m.value >= factor, m.value % factor > 0]
                        j_shape = [factor, m.value % factor]
                        j_cond_expr = [(m_block_idx + 1) * factor <= m, m_block_idx + 1 == m_grid_dim]
                        if i_cond[i] and j_cond[j]:
                            shape = [i_shape[i], j_shape[j]]
                            subtask_name = f'{task.name}.f{factor}.d{block_dim}.s{shape[0]}x{shape[1]}.block'
                            subtask, args = self.get_subtask(subtask_name, task, match, factor, shape, block_dim, n_block_idx, m_block_idx, func_params)
                            submodule = implement(subtask)
                            ir_module.include(submodule)
                            cond = convert(True)
                            if i_cond[1 - i]:
                                cond = And(cond, i_cond_expr[i])
                            if j_cond[1 - i]:
                                cond = And(cond, j_cond_expr[j])
                            stmts = []
                            stmts.append(IfStmt(cond))
                            stmts.append(EvaluateStmt(Call(ir_module.lookup_var(subtask_name), args)))
                            func_body_stmts.append(concat_stmts(stmts))
                func_body = concat_stmts(func_body_stmts)
                func = Function(func_name, func_params, func_body, VoidType(), func_local_vars, {'worker': Grid(convert(grid_dim), convert(block_dim))})
                func_group.append(func)
        ir_module.add(func_name, func_group)
        return ir_module
