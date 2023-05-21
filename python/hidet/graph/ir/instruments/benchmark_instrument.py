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
from typing import List, Optional, Dict, Tuple
import os
import numpy as np

from hidet.runtime import CompiledFunction
from hidet.graph.ir.flow_graph import FlowGraph, Operator, Tensor, GraphForwardInstrument, SizeVar


class GraphForwardBenchmarkInstrument(GraphForwardInstrument):
    _template = '{:>5} {:>25}  {:70} {:20} {:>15} {:>10}'

    def __init__(self, output_dir='./outs/benchmark', print_summary=False, warmup=3, number=10, repeat=3):
        self.output_dir: str = output_dir
        self.print_summary: bool = print_summary
        self.warmup: int = warmup
        self.number: int = number
        self.repeat: int = repeat

        self.benchmarking: bool = False
        self.run_dir: Optional[str] = None
        self.latency_list: List[Tuple[Operator, float, float]] = []
        self.reset()

    def reset(self):
        self.benchmarking = False
        self.run_dir = None
        self.latency_list = []

    def get_run_dir(self) -> str:
        # find a unique directory under given output_dir
        idx = 0
        while os.path.exists(os.path.join(self.output_dir, str(idx))):
            idx += 1
        return os.path.join(self.output_dir, str(idx))

    def before_graph(self, graph: FlowGraph, inputs: List[Tensor]) -> None:
        if any(tensor.is_symbolic() for tensor in inputs):
            return

        self.benchmarking = True
        self.run_dir = self.get_run_dir()
        self.latency_list = []
        os.makedirs(self.run_dir, exist_ok=True)

    def after_operator(
        self, op: Operator, inputs: List[Tensor], shape_map: Dict[SizeVar, int], outputs: List[Tensor]
    ) -> None:
        if not self.benchmarking:
            return

        task_func: CompiledFunction = op.task_func
        latency: List[float] = task_func.profile(
            *inputs, *outputs, warmup=self.warmup, number=self.number, repeat=self.repeat
        )
        self.latency_list.append((op, float(np.median(latency)), float(np.std(latency))))

    def after_graph(self, graph: FlowGraph, inputs: List[Tensor], outputs: List[Tensor]) -> None:
        if not self.benchmarking:
            return
        head_line = self._template.format('Index', 'Operator', 'Inputs', 'Outputs', 'Latency(us)', 'Std(us)')
        outer_sep = '-' * len(head_line)
        inner_sep = self._template.format(*['-' * v for v in [5, 25, 70, 20, 15, 10]])
        lines = []
        lines.append(outer_sep)
        lines.append(head_line)
        lines.append(inner_sep)
        total_latency: float = sum(latency for _, latency, _ in self.latency_list)
        total_std: float = float(np.sqrt(sum(std**2 for _, _, std in self.latency_list)))
        for idx, (op, latency, std) in enumerate(self.latency_list):
            op_idx = '{}'.format(idx)
            op_name = '{}'.format(op.name)
            inputs = ', '.join([(tensor.dtype.short_name + str(tensor.shape)).replace(' ', '') for tensor in op.inputs])
            outputs = ', '.join(
                [(tensor.dtype.short_name + str(tensor.shape)).replace(' ', '') for tensor in op.outputs]
            )
            lines.append(
                self._template.format(
                    op_idx, op_name, inputs, outputs, '{:10.3f}'.format(latency * 1000), '{:10.3f}'.format(std * 1000)
                )
            )
        lines.append(inner_sep)
        lines.append(
            self._template.format(
                '', '', '', '', '{:10.3f}'.format(total_latency * 1000), '{:10.3f}'.format(total_std * 1000)
            )
        )
        lines.append(outer_sep)
        summary: str = '\n'.join(lines)
        summary_file: str = os.path.join(self.run_dir, 'summary.txt')
        with open(summary_file, 'w') as f:
            f.write(summary)
        if self.print_summary:
            print(summary)
        print('Benchmark summary has been saved to: {}'.format(os.path.realpath(summary_file)))
        self.reset()
