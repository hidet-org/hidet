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
from typing import List

from hidet.graph.flow_graph import FlowGraph
from hidet.graph.transforms import GraphPass
from .graph_patterns import SubgraphRewriteRule
from .subgraph_rewrite import SubgraphRewritePass


class SelectiveQuantizePass(GraphPass):
    def process_graph(self, graph: FlowGraph) -> FlowGraph:
        rewrite_patterns: List[SubgraphRewriteRule] = self.current_context().configs['quantize_patterns']
        graph = SubgraphRewritePass(rewrite_patterns)(graph)
        return graph


def selective_quantize_pass() -> GraphPass:
    return SelectiveQuantizePass()
