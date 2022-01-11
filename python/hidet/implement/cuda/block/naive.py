from typing import Mapping, List, Tuple

from hidet.ir.node import Node
from hidet.ir.type import scalar_type, tensor_type, TensorType
from hidet.ir.expr import Expr, Var, IntVar, Call, convert, Constant, Axis, TensorElement
from hidet.ir.stmt import LetStmt, IfStmt, EvaluateStmt, concat_stmts, SeqStmt, BufferStoreStmt, ForStmt
from hidet.ir.task import Task, Grid, ThreadBlock, Thread
from hidet.ir.func import IRModule, Function, FunctionGroup
from hidet.ir.dialects.compute import TensorInput, ScalarInput, TensorCompute, compute, scalar_input
from hidet.ir.dialects.pattern import TaskPattern, TensorComputePattern, ScalarExprPattern
from hidet.ir.dialects.lowlevel import VoidType, Address
from hidet.ir.functors import rewrite, infer_type, collect, simplify
from hidet.implement.implementer import Implementer, implement, register_impl


def product(lst: List[Constant]):
    if len(lst) == 0:
        return convert(1)
    s = lst[0].value
    for v in lst[1:]:
        s = s * v.value
    return s


@register_impl('cuda_block_naive_implementer')
class CudaBlockNaiveImplementer(Implementer):
    def __init__(self):
        self.block_dim = Constant(None, dtype=scalar_type('int32'))
        self.computation = TensorComputePattern(rank=None, allow_dynamic_axis=False)
        self.pattern = TaskPattern(
            compute_pattern=self.computation,
            required_params=[],
            required_param_types=[],
            allow_tensor_extra_params=True,
            worker=ThreadBlock(block_dim=self.block_dim)
        )

    def priority(self) -> int:
        return 0

    def task_pattern(self) -> TaskPattern:
        return self.pattern

    def get_subtask(self,
                    task: Task,
                    match: Mapping[Node, Node],
                    subtask_name: str,
                    indices: List[Var],
                    func_params: List[Var]) -> Tuple[Task, List[Expr]]:
        param2var = {param: var for param, var in zip(task.params, func_params)}
        block_dim: Constant = match[self.block_dim]
        computation: TensorCompute = match[self.computation]
        rank: int = len(computation.shape)
        value: Expr = computation.value

        indices_params = [scalar_input('i', 'int32') for i in range(rank)]
        orig_input_params = [p for p in task.params if p is not task.compute]
        subtask_computation = rewrite(value, {axis: idx for axis, idx in zip(computation.axes, indices_params)})
        subtask_params = indices_params + orig_input_params + [subtask_computation]

        param2type = {param: param_type for param, param_type in zip(task.params, task.params_type)}
        param2type.update({index: scalar_type('int32') for index in indices_params})
        param2type[subtask_computation] = infer_type(subtask_computation)
        subtask_params_type = [param2type[p] for p in subtask_params]

        out_arg = TensorElement(param2var[task.compute], indices)
        subtask_args = indices + [param2var[p] for p in orig_input_params] + [out_arg]
        return Task(subtask_name, subtask_computation, subtask_params, subtask_params_type, Thread()), subtask_args

    def implement(self, task: Task, match: Mapping[Node, Node]) -> IRModule:
        assert isinstance(task.worker, ThreadBlock)
        block_dim: Constant = match[self.block_dim]
        computation: TensorCompute = match[self.computation]

        rank = len(computation.shape)
        shape: List[Constant] = computation.shape

        task_size: int = product(computation.shape)
        block_size = block_dim.value
        num_rounds = (task_size + block_size - 1) // block_size

        thread_idx = IntVar('threadIdx.x')
        round_idx = Axis(num_rounds, hint='round_idx')
        task_idx = IntVar('task_idx')

        stmts = [ForStmt(round_idx),
                 LetStmt(task_idx, round_idx * block_size + thread_idx),
                 IfStmt(task_idx < task_size)]
        indices = [IntVar() for _ in range(rank)]
        for i in range(rank):
            stmts.append(LetStmt(indices[i], (task_idx / product(shape[i + 1:])) % shape[i]))

        subtask_name = f'{task.name}.thread'
        func_params: List[Var] = [Var(param.name, param_type) for param, param_type in zip(task.params, task.params_type)]
        subtask, args = self.get_subtask(task, match, subtask_name, indices, func_params)
        submodule = implement(subtask)

        ir_module = IRModule()
        ir_module.include(submodule)
        stmts.append(EvaluateStmt(Call(ir_module.lookup_var(subtask_name), args)))

        func_body = concat_stmts(stmts)
        func = Function(task.name, func_params, func_body, VoidType(), [], {'worker': task.worker})
        ir_module.add(task.name, func)
        return ir_module

