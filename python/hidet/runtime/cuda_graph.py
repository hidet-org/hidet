from typing import List, Optional
from hidet.tos.tensor import Tensor


class ExecutionContext:
    def __init__(self):
        self.op_inputs: List[List[Tensor]] = []
        self.op_outputs: List[List[Tensor]] = []


class CudaGraph:
    def __init__(self):
        self.ctx: Optional[ExecutionContext] = None

    def __call__(self, *args):
        pass

    def __del__(self):
        pass

    def forward(self, *args):
        pass


def create_cuda_graph(flow_graph) -> CudaGraph:
    pass
