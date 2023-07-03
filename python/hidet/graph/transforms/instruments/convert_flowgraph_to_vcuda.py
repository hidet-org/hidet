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

from hidet import utils
from hidet.graph.flow_graph import FlowGraph

from .base import GraphPassInstrument


class ConvertGraphToVCuda(GraphPassInstrument):
    def before_all_passes(self, graph: FlowGraph):
        graph._to_vcuda()

    def after_all_passes(self, graph: FlowGraph) -> None:
        graph._from_vcuda()
