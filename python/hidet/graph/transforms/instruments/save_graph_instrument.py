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
from hidet.graph.ir.flow_graph import FlowGraph

from .base import GraphPassInstrument


class SaveGraphInstrument(GraphPassInstrument):
    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        self.index = 0
        os.makedirs(out_dir, exist_ok=True)

    def before_all_passes(self, graph: FlowGraph):
        # first clean all json starting with indices
        for fname in os.listdir(self.out_dir):
            fpath = os.path.join(self.out_dir, fname)
            parts = fname.split('_')
            if os.path.isfile(fpath) and len(parts) > 1 and parts[0].isdigit() and fname.endswith('.json'):
                os.remove(fpath)
        with open(os.path.join(self.out_dir, '0_Origin.json'), 'w') as f:
            utils.netron.dump(graph, f)
            self.index += 1

    def after_pass(self, pass_name: str, graph: FlowGraph):
        with open(os.path.join(self.out_dir, '{}_{}.json'.format(self.index, pass_name)), 'w') as f:
            utils.netron.dump(graph, f)
            self.index += 1
