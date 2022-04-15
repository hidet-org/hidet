from typing import Union, List, Optional, Dict, Sequence
import os
import time
from hidet.tos.ir.graph import FlowGraph, Operator, Tensor
from hidet import utils


class InstrumentContext:
    def before_all_passes(self, graph: FlowGraph):
        pass

    def before_pass(self, pass_name: str, graph: FlowGraph):
        pass

    def after_pass(self, pass_name: str, graph: FlowGraph):
        pass

    def after_all_passes(self, graph: FlowGraph):
        pass


class SaveGraphInstrument(InstrumentContext):
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


class ProfileInstrument(InstrumentContext):
    def __init__(self, log_file: str, print_stdout: bool = False):
        dirname = os.path.dirname(log_file)
        os.makedirs(dirname, exist_ok=True)
        self.log_file = log_file
        self.print_stdout = print_stdout
        self.start_time: Dict[str, float] = {}

    def before_all_passes(self, graph: FlowGraph):
        # clear file contents
        with open(self.log_file, 'w'):
            pass

    def before_pass(self, pass_name: str, graph: FlowGraph):
        self.start_time[pass_name] = time.time()

    def after_pass(self, pass_name: str, graph: FlowGraph):
        elapsed_time = time.time() - self.start_time[pass_name]
        with open(self.log_file, 'a') as f:
            f.write('{:>50} {:.3f} seconds\n'.format(pass_name, elapsed_time))
        if self.print_stdout:
            print('{:>50} {} seconds'.format(pass_name, utils.py.green(elapsed_time, '{:.3f}')))


class PassContext:
    stack: List['PassContext'] = []

    def __init__(self, instruments: Sequence[InstrumentContext] = (), verbose: bool = False):
        self.instruments: Sequence[InstrumentContext] = instruments
        self.verbose: bool = verbose

    @classmethod
    def current(cls):
        return cls.stack[-1]

    def __enter__(self):
        self.stack.append(self)

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert len(self.stack) > 0 and self.stack[-1] is self
        self.stack.pop()


# add top-level pass context
PassContext.stack.append(PassContext())


class GraphPass:
    def __init__(self):
        self.name = self.__class__.__name__

    def __call__(self, graph: FlowGraph) -> FlowGraph:
        ctx = PassContext.current()
        for inst in ctx.instruments:
            inst.before_pass(self.name, graph)
        graph = self.process_graph(graph)
        for inst in reversed(ctx.instruments):
            inst.after_pass(self.name, graph)
        return graph

    @staticmethod
    def current_context() -> PassContext:
        return PassContext.current()

    def process_graph(self, graph: FlowGraph) -> FlowGraph:
        raise NotImplementedError()
