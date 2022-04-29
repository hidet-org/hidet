from typing import List, Sequence
import logging

from hidet.tos.ir.graph import FlowGraph
from .instruments import GraphPassInstrument

logger = logging.Logger(name='hidet.tos.transforms', level=logging.INFO)
logger.addHandler(logging.StreamHandler())


class PassContext:
    stack: List['PassContext'] = []

    def __init__(self, instruments: Sequence[GraphPassInstrument] = (), verbose: bool = False):
        self.instruments: Sequence[GraphPassInstrument] = instruments
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
