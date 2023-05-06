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
from typing import Optional, Dict, List, Tuple
import os
import time

from hidet import utils
from hidet.ir.func import IRModule

from .base import PassInstrument


class ProfileInstrument(PassInstrument):
    def __init__(self, log_file: Optional[str] = None, print_stdout: bool = False):
        if log_file:
            dirname = os.path.dirname(log_file)
            os.makedirs(dirname, exist_ok=True)
        self.log_file = log_file
        self.print_stdout = print_stdout
        self.start_time: Dict[str, float] = {}
        self.elapsed_time: List[Tuple[str, float]] = []

    def before_all_passes(self, ir_module: IRModule):
        if self.log_file:
            # clear file contents
            with open(self.log_file, 'w'):
                pass

    def before_pass(self, pass_name: str, ir_module: IRModule):
        self.start_time[pass_name] = time.time()
        if self.print_stdout:
            print('{:>50} started...'.format(pass_name))

    def after_pass(self, pass_name: str, ir_module: IRModule):
        elapsed_time = time.time() - self.start_time[pass_name]
        self.elapsed_time.append((pass_name, elapsed_time))
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write('{:>50} {:.3f} seconds\n'.format(pass_name, elapsed_time))
        if self.print_stdout:
            print('{:>50} {} seconds'.format(pass_name, utils.py.green(elapsed_time, '{:.3f}')))

    def after_all_passes(self, ir_module: IRModule):
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write('{:>50} {:.3f} seconds\n'.format('total', sum([x[1] for x in self.elapsed_time])))
                f.write('\n')
                f.write('descending order:\n')
                self.elapsed_time.sort(key=lambda x: x[1], reverse=True)
                for pass_name, elapsed_time in self.elapsed_time:
                    f.write('{:>50} {:.3f} seconds\n'.format(pass_name, elapsed_time))
        if self.print_stdout:
            print('{:>50} {} seconds'.format('total', utils.py.green(sum([x[1] for x in self.elapsed_time]), '{:.3f}')))
