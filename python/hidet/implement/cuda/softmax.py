from typing import Mapping, Any

from hidet.implement.implementer import Implementer, register_impl
from hidet.ir import IRModule
from hidet.ir.dialects.pattern import TaskPattern
from hidet.ir.node import Node
from hidet.ir.task import Task


@register_impl('cuda_grid_static_matmul_implementer')
class CudaGridStaticMatmulImplementer(Implementer):
    def __init__(self):
        super().__init__()

    def priority(self) -> int:
        return 1

    def task_pattern(self) -> TaskPattern:
        pass

    def implement(self, task: Task, match: Mapping[Node, Any]) -> IRModule:
        pass
