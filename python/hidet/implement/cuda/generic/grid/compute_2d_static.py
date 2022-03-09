from typing import Mapping, List, Dict, Union

from hidet.ir.node import Node
from hidet.ir.type import scalar_type, tensor_type, TensorType
from hidet.ir.expr import Expr, Var, Call, And, convert, Constant, TensorElement, var
from hidet.ir.stmt import IfStmt, EvaluateStmt, concat_stmts, SeqStmt, BufferStoreStmt, LetStmt
from hidet.ir.task import Task, Grid, ThreadBlock, Thread
from hidet.ir.func import IRModule, Function, FunctionGroup
from hidet.ir.dialects.compute import TensorInput, ScalarInput, TensorCompute, compute, scalar_input
from hidet.ir.dialects.pattern import TaskPattern, TensorComputePattern, ScalarExprPattern
from hidet.ir.dialects.lowlevel import VoidType, Address
from hidet.ir.functors import rewrite, infer_type, collect, simplify, coefficients
from hidet.implement.implementer import Implementer, implement, register_impl, NotSupportedError
from hidet.implement.search_space import SpaceChoice, ProductSpace, AtomSpace
from hidet.ir.builders import FunctionBuilder, StmtBuilder
from hidet.ir.primitives import block_idx


@register_impl('cuda_grid_split_implementer')
class CudaGridSplitImplementer(Implementer):
    def __init__(self):
        self.n = Constant(None, dtype=scalar_type('int32'))
        self.m = Constant(None, dtype=scalar_type('int32'))
        self.axes = [var(), var()]
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

    def priority(self) -> int:
        return 1

    def task_pattern(self) -> TaskPattern:
        return self.pattern

    def get_subtask(self,
                    subtask_name,
                    task: Task,
                    match: Mapping[Node, Node],
                    block_task: List[int],
                    sub_shape: List[int],
                    block_dim: int,
                    n_block_idx: Var,
                    m_block_idx: Var,
                    func_params: List[Var]
                    ) -> Task:
        param2type = {param: param_type for param, param_type in zip(task.params, task.params_type)}
        param2var = {param: arg for param, arg in zip(task.params, func_params)}
        axes: List[Var] = [match[axis] for axis in self.axes]
        value: Expr = match[self.value]
        subtask_axes = [var(), var()]
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
                    raise NotSupportedError("do not consider not-used tensor input")
                if len(tes) > 1:
                    raise NotSupportedError("do not support multiple access to the same tensor input as far as now")
                te = tes[0]
                indices = te.indices
                arg_indices = []
                for idx_expr in indices:
                    coeffs = coefficients(idx_expr, axes)
                    if any(sum(order) > 1 for order in coeffs):
                        raise NotSupportedError("do not support high order indexing e.g., A[i * i][j], A[i * j][k]")
                    for order, coeff in coeffs.items():
                        if sum(order) > 0 and any(v not in task.params for v in collect(coeff, [Var, ScalarInput])):
                            raise NotSupportedError("The coefficient is dependent on inner var, e.g., A[i * reduce_axis], where reduce_axis is a inner var")
                    s = convert(0)
                    for order in coeffs:
                        for i in range(2):
                            if order[i] == 1:
                                s = s + coeffs[order] * outer_axes[i] * block_task[i]
                    arg_indices.append(s)
                rmap[te] = rewrite(te, {a: b for a, b in zip(axes, subtask_axes)})
                subtask_param2arg[ti] = Address(TensorElement(param2var[ti], arg_indices))  # update
            subtask_value = rewrite(value, rmap)

            subtask_compute = TensorCompute('out', sub_shape, subtask_axes, subtask_value)
            input_params = collect(subtask_value, (ScalarInput, TensorInput))
            subtask_params = input_params + [subtask_compute]
            param2type[subtask_compute] = param2type[task.compute]
            subtask_params_type = [param2type[param] for param in subtask_params]

            out_arg = Address(TensorElement(param2var[task.compute], [n_block_idx * block_task[0], m_block_idx * block_task[1]]))
            args = [subtask_param2arg[param] for param in input_params] + [out_arg]

            return Task(subtask_name, subtask_compute, subtask_params, subtask_params_type, ThreadBlock(block_dim)), args

        except NotSupportedError:
            # fallback subtask with block index
            outer_axes_params = [scalar_input('i_o', 'int32'), scalar_input('j_o', 'int32')]
            sub_compute = compute('out', sub_shape,
                                  lambda i, j: rewrite(value, {old_axis: inner_axis + outer_axis * block_task[idx]
                                                               for idx, old_axis, inner_axis, outer_axis in zip(range(2), axes, [i, j], outer_axes_params)}))
            output_param = sub_compute

            other_params = collect(value, (ScalarInput, TensorInput))
            params = outer_axes_params + other_params + [output_param]
            task_ret_type: TensorType = param2type[task.compute]
            subtask_ret_type: TensorType = tensor_type(task_ret_type.scope, task_ret_type.scalar_type, block_task, task_ret_type.layout)
            params_type = [scalar_type('int32'), scalar_type('int32')] + [param2type[param] for param in other_params] + [subtask_ret_type]

            # out_arg = param2var[task.compute][n_block_idx: n_block_idx + sub_shape[0], m_block_idx: m_block_idx + sub_shape[1]]
            out_arg = Address(TensorElement(param2var[task.compute], [n_block_idx * block_task[0], m_block_idx * block_task[1]]))
            args = [n_block_idx, m_block_idx] + [param2var[param] for param in other_params] + [out_arg]

            return Task(subtask_name, sub_compute, params, params_type, ThreadBlock(block_dim)), args

    def implement(self, task: Task, match: Mapping[Node, Node]) -> IRModule:
        # space = ProductSpace(
        #     'space', [
        #         AtomSpace('factor', [16, 32, 48, 64, 96, 128, 192, 256]),
        #         AtomSpace('block_size', [256, 512, 1024])
        #     ]
        # )
        # space = ProductSpace(
        #     'space', [
        #         AtomSpace('block_task', [[128, 128], [16, 16]]),
        #         AtomSpace('block_size', [256, 32])
        #     ]
        # )
        space = ProductSpace(
            'space', [
                AtomSpace('block_task', [[128, 128]]),
                AtomSpace('block_size', [256])
            ]
        )
        space_size = len(space)
        ir_module = IRModule(task=task)
        for i in range(space_size):
            choice = space[i]
            try:
                choice_module = self.implement_for_choice(task, match, choice)
            except NotSupportedError as e:
                continue
            else:
                ir_module.include(choice_module)
        return ir_module

    def implement_for_choice(self, task: Task, match: Mapping[Node, Node], choice: SpaceChoice):
        assert isinstance(task.compute, TensorCompute)
        task_n = int(match[self.n])
        task_m = int(match[self.m])

        block_task = choice.block_task.value
        block_size = choice.block_size.value
        grid_n = (task_n + block_task[0] - 1) // block_task[0]
        grid_m = (task_m + block_task[1] - 1) // block_task[1]
        grid_size = grid_n * grid_m

        ir_module = IRModule()
        with FunctionBuilder(task.name + '.grid') as fb:
            blockIdx = block_idx()
            fb.extend_params([Var(param.name, param_type) for param, param_type in zip(task.params, task.params_type)])
            n_block_idx = var('n_block_idx')
            m_block_idx = var('m_block_idx')

            sb = StmtBuilder()
            sb.enter_body(LetStmt(n_block_idx, blockIdx / grid_m))
            sb.enter_body(LetStmt(m_block_idx, blockIdx % grid_m))
            for i in range(2):
                i_cond = [task_n >= block_task[0], task_n % block_task[0] > 0]
                i_shape = [block_task[0], task_n % block_task[0]]
                i_cond_expr = [(n_block_idx + 1) * block_task[0] <= task_n, n_block_idx + 1 == grid_n]
                for j in range(2):
                    j_cond = [task_m >= block_task[1], task_m % block_task[1] > 0]
                    j_shape = [block_task[1], task_m % block_task[1]]
                    j_cond_expr = [(m_block_idx + 1) * block_task[1] <= task_m, m_block_idx + 1 == grid_m]
                    if i_cond[i] and j_cond[j]:
                        shape = [i_shape[i], j_shape[j]]
                        subtask_name = f'{task.name}_bt{block_task[0]}x{block_task[1]}_bsz{block_size}_s{shape[0]}x{shape[1]}_block'
                        subtask, args = self.get_subtask(subtask_name, task, match, block_task, shape, block_size, n_block_idx, m_block_idx, fb.params)
                        submodule = implement(subtask)
                        ir_module.include(submodule)
                        cond = convert(True)
                        if i_cond[1 - i]:
                            cond = And(cond, i_cond_expr[i])
                        if j_cond[1 - i]:
                            cond = And(cond, j_cond_expr[j])
                        with sb.if_then(cond):
                            sb.append(EvaluateStmt(Call(ir_module.lookup_var(subtask_name), args)))

            sb.exit_body()  # let n_block_idx
            sb.exit_body()  # let m_block_idx
            body = sb.finish()
            fb.set_body(body)
            fb.extend_attrs({'worker': Grid(convert(grid_size), convert(block_size)),
                             'label': f'block_task-{block_task[0]}x{block_task[1]}-block_size-{block_size}'})
        func = fb.get()
        ir_module.add(func.name, func)
        return ir_module
