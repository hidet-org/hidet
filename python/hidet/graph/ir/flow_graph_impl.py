from typing import List, Optional, Dict
import os
import numpy as np
from hidet.ir.type import data_type
from .flow_graph import FlowGraph, Operator, Tensor, GraphForwardInstrument


class GraphForwardDebugInstrument(GraphForwardInstrument):
    _template = '{:>5} {:>25} {:>3}   {:<25} {:>8} {:>8} {:>8} {:>10} {:>10} {:>10} {:>10}'

    def __init__(self, output_dir='./outs/debug', print_stdout=False):
        self.output_dir: str = output_dir
        self.print_stdout: bool = print_stdout

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
                '{}{}'.format(data_type(tensor.dtype).short_name, list(tensor.shape)),
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
                '{}{}'.format(data_type(tensor.dtype).short_name, list(tensor.shape)),
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
        if self.print_stdout:
            with open(self.summary_file, 'r') as f:
                print(f.read())
        print('Debug summary has been saved to: {}'.format(self.summary_file))
        self.reset()


class GraphForwardBenchmarkInstrument(GraphForwardInstrument):
    pass
