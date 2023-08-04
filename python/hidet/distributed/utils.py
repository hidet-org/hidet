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

import os

import hidet
from hidet.graph import FlowGraph, Tensor
from hidet.graph.graph_utils.functors import GraphCloneRewriter
from hidet.runtime.device import Device, instantiate_device


class GraphMover(GraphCloneRewriter):
    def __init__(self, device: Device):
        self.device = device
        self.current_device = None
        self.memo = {}

    def visit_FlowGraph(self, graph: FlowGraph):
        self.current_device = None
        self.memo = {}
        self.input_map = {}
        outputs = [self.visit(output) for output in graph.outputs]
        inputs = [self.input_map[i] for i in graph.inputs]
        return FlowGraph(outputs, inputs)

    # full, ones, zeros, etc...

    def visit_Tensor(self, tensor: Tensor):
        if self.current_device is None:
            self.current_device = tensor.device
        elif self.current_device != tensor.device:
            raise RuntimeError("All tensors in the FlowGraph are supposed to be placed on the same device")
        if tensor in self.memo:
            return self.memo[tensor]
        if tensor.trace is None:
            if tensor.is_symbolic():  # inputs
                self.memo[tensor] = Tensor(tensor.shape, tensor.dtype, self.device, None, tensor.layout)
                self.input_map[tensor] = self.memo[tensor]
            else:  # parameters
                self.memo[tensor] = tensor.to(device=self.device)
        else:
            self(tensor.trace[0])
        return self.memo[tensor]


def copy_flowgraph_to(graph: FlowGraph, device):
    device = instantiate_device(device)
    mover = GraphMover(device)
    return mover(graph)


def load_partition(out_dir: str, rank: int, device='cuda'):
    graph = hidet.load_graph(os.path.join(out_dir, f"part{rank}.graph"))
    return copy_flowgraph_to(graph, device)
