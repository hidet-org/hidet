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
from .flow_graph import FlowGraph, Operator, Tensor, GraphForwardInstrument


class GraphForwardDebugInstrument(GraphForwardInstrument):
    _template = '{:>5} {:>25} {:>3}   {:<25} {:>8} {:>8} {:>8} {:>10} {:>10} {:>10} {:>10}'

    def __init__(self, output_dir='./outs/debug', print_summary=False):
        self.output_dir: str = output_dir
        self.print_summary: bool = print_summary

        self.debugging: bool = False
        self.run_dir: Optional[str] = None
        self.summary_file: Optional[str] = None
        self.operator_idx: int = 0
        self.reset()

    def reset(self):
        self.debugging = False
        self.run_dir = None
        self.summary_file = None
        self.operator_idx = 0

    def get_run_dir(self) -> str:
        # find a unique directory under given output_dir
        idx = 0
        while os.path.exists(os.path.join(self.output_dir, str(idx))):
            idx += 1
        return os.path.join(self.output_dir, str(idx))

    def before_graph(self, graph: FlowGraph, inputs: List[Tensor]) -> None:
        from hidet.utils import netron

        if any(tensor.is_symbolic() for tensor in inputs):
            return

        self.debugging = True
        self.run_dir = self.get_run_dir()
        self.summary_file: str = os.path.join(self.run_dir, 'summary.txt')
        self.operator_idx = 0
        os.makedirs(self.run_dir, exist_ok=True)

        with open(os.path.join(self.run_dir, 'graph.json'), 'w') as f:
            netron.dump(graph, f)

        with open(self.summary_file, 'w') as f:
            head = self._template.format(
                'Index', 'Operator', 'Arg', 'Shape', 'NaN', 'Inf', 'Zero', 'Min', 'Max', 'Mean', 'Std'
            )
            top_sep = '-' * len(head)
            bot_sep = self._template.format(*['-' * extent for extent in [5, 25, 3, 25, 8, 8, 8, 10, 10, 10, 10]])
            f.write(''.join([top_sep, '\n', head, '\n', bot_sep, '\n']))

    @staticmethod
    def tensor_stats(array: np.ndarray) -> Dict[str, str]:
        double_array = array.astype(np.float64)
        return {
            'nan': '{:8d}'.format(np.count_nonzero(np.isnan(array))),
            'inf': '{:8d}'.format(np.count_nonzero(np.isinf(array))),
            'zero': '{:8d}'.format(np.count_nonzero(array == 0)),
            'min': '{:8.2e}'.format(np.min(double_array)),
            'max': '{:8.2e}'.format(np.max(double_array)),
            'mean': '{:8.2e}'.format(np.mean(double_array)),
            'std': '{:8.2e}'.format(np.std(double_array)),
        }

    def before_operator(self, op: Operator, inputs: List[Tensor]) -> None:
        if not self.debugging:
            return

        lines = []
        for idx, tensor in enumerate(inputs):
            array: np.ndarray = tensor.numpy(share_mem=False)
            stats = self.tensor_stats(array)
            if idx == 0:
                op_idx = str(self.operator_idx)
                op_name = op.name
            else:
                op_idx = op_name = ''
            line = self._template.format(
                op_idx,
                op_name,
                'x{}'.format(idx),
                '{}{}'.format(tensor.dtype.short_name, list(tensor.shape)),
                stats['nan'],
                stats['inf'],
                stats['zero'],
                stats['min'],
                stats['max'],
                stats['mean'],
                stats['std'],
            )
            lines.append(line)
        with open(self.summary_file, 'a') as f:
            f.write('\n'.join(lines) + '\n')

    def after_operator(self, op: Operator, inputs: List[Tensor], outputs: List[Tensor]) -> None:
        if not self.debugging:
            return

        lines = []
        found_abnormal = False
        for idx, tensor in enumerate(outputs):
            array: np.ndarray = tensor.numpy(share_mem=False)
            stats = self.tensor_stats(array)
            op_idx = op_name = ''
            line = self._template.format(
                op_idx,
                op_name,
                'y{}'.format(idx),
                '{}{}'.format(tensor.dtype.short_name, list(tensor.shape)),
                stats['nan'],
                stats['inf'],
                stats['zero'],
                stats['min'],
                stats['max'],
                stats['mean'],
                stats['std'],
            )
            lines.append(line)
            found_abnormal = found_abnormal or int(stats['nan']) > 0 or int(stats['inf']) > 0

        with open(self.summary_file, 'a') as f:
            f.write('\n'.join(lines) + '\n')

        if found_abnormal:
            arrays = [tensor.numpy(share_mem=False) for tensor in inputs + outputs]
            names = ['x{}'.format(idx) for idx in range(len(inputs))] + [
                'y{}'.format(idx) for idx in range(len(outputs))
            ]
            np.savez(os.path.join(self.run_dir, 'abnormal_in_outs.npz'), **dict(zip(names, arrays)))
            with open(os.path.join(self.run_dir, '{}.txt'.format(op.name)), 'w') as f:
                f.write('Operator:\n{}\n'.format(op))
                f.write('Task:\n{}\n'.format(op.task))
            msg = 'Found nan/inf in outputs of operator {}. Its inputs and outputs are dumped to: {}'.format(
                op.name, self.run_dir
            )
            self.reset()
            raise ValueError(msg)

        self.operator_idx += 1

    def after_graph(self, graph: FlowGraph, inputs: List[Tensor], outputs: List[Tensor]) -> None:
        if self.print_summary:
            with open(self.summary_file, 'r') as f:
                print(f.read())
        print('Debug summary has been saved to: {}'.format(os.path.realpath(self.summary_file)))
        self.reset()


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

    def after_operator(self, op: Operator, inputs: List[Tensor], outputs: List[Tensor]) -> None:
        if not self.benchmarking:
            return

        if op.task_func is None:
            op.build_task_func()
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
