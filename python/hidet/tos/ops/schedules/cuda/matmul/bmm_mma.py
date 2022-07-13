from typing import List, Tuple, Union, Optional

import os
from hidet.ir.builders import FunctionBuilder, StmtBuilder
from hidet.ir.dialects.lowlevel import TensorPointerType, PointerType
from hidet.ir.expr import Var, And, Equal, Cast, if_then_else, convert, Expr
from hidet.ir.func import IRModule
from hidet.ir.functors import simplify_to_int
from hidet.ir.layout import TaskLayout, DataLayout, StridesLayout
from hidet.ir.primitives import syncthreads, thread_idx, block_idx
from hidet.ir.stmt import AssignStmt, BufferStoreStmt, IfStmt
from hidet.ir.type import scalar_type, tensor_type, ScalarType
from hidet.ir.task import TaskContext
from hidet.utils import cuda
from hidet.tos.ops.definitions.matmul.matmul import MatmulTask
from hidet.tos.ops.schedules.resolve import resolve_ir_modules
from hidet.tos.ops.schedules.common import params_from_task, Schedule, NotSupportedError


class MatmulMmaSchedule(Schedule):

    def keys(self) -> List[Tuple[str, Union[int, float, str]]]:
        pass

    def derived_keys(self) -> List[Tuple[str, Union[int, float, str]]]:
        pass


