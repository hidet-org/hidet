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
from hidet.graph.flow_graph import FlowGraph
from hidet.graph.graph_utils.functors import graph_collect
from .base import GraphPassInstrument


class ConvertGraphToVGPU(GraphPassInstrument):
    def __init__(self, enable: bool, target='cuda'):
        super().__init__()
        self.enable = enable
        self.applied = False
        self.target = target
        assert target in ['cuda', 'hip']

        # passes may take up to 2x memory (80% of total memory)
        self.threshold = 0.4

    def before_all_passes(self, graph: FlowGraph):
        if not self.should_enable(graph):
            return
        if self.target == 'cuda':
            graph.vcuda_()
        else:
            graph.vhip_()
        self.applied = True

    def after_all_passes(self, graph: FlowGraph) -> None:
        if self.applied:
            if self.target == 'cuda':
                graph.cuda_()
            else:
                graph.hip_()
            self.applied = False

    def should_enable(self, graph):
        from hidet.graph import Tensor

        if self.enable is None:
            tensors = graph_collect(graph, Tensor)
            graph_size = 0
            for t in tensors:
                if t.storage is not None and (t.storage.device.is_cuda() or t.storage.device.is_hip()):
                    graph_size += t.storage.num_bytes

            if self.target == 'cuda':
                from hidet.cuda import current_device, properties
            else:
                from hidet.hip import current_device, properties

            return graph_size / properties(current_device()).totalGlobalMem > self.threshold
        else:
            return self.enable
