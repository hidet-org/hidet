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
import time
from typing import Optional, Dict

from hidet import utils
from hidet.graph.ir.flow_graph import FlowGraph

from .base import GraphPassInstrument


class ProfileInstrument(GraphPassInstrument):
    def __init__(self, log_file: Optional[str] = None, print_stdout: bool = False):
        if log_file:
            dirname = os.path.dirname(log_file)
            os.makedirs(dirname, exist_ok=True)
        self.log_file = log_file
        self.print_stdout = print_stdout
        self.start_time: Dict[str, float] = {}

    def before_all_passes(self, graph: FlowGraph):
        if self.log_file:
            # clear file contents
            with open(self.log_file, 'w'):
                pass

    def before_pass(self, pass_name: str, graph: FlowGraph):
        self.start_time[pass_name] = time.time()
        if self.print_stdout:
            print('{:>50} started...'.format(pass_name))

    def after_pass(self, pass_name: str, graph: FlowGraph):
        elapsed_time = time.time() - self.start_time[pass_name]
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write('{:>50} {:.3f} seconds\n'.format(pass_name, elapsed_time))
        if self.print_stdout:
            print('{:>50} {} seconds'.format(pass_name, utils.py.green(elapsed_time, '{:.3f}')))
