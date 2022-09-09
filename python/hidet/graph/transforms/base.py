from __future__ import annotations
from typing import List, Sequence, Optional, Dict, Any
import logging

from hidet.graph.ir.flow_graph import FlowGraph
from .instruments import GraphPassInstrument

logger = logging.Logger(name='hidet.graph.transforms', level=logging.INFO)
logger.addHandler(logging.StreamHandler())


class PassContext:
    _stack: List['PassContext'] = []

    def __init__(self):
        self.instruments: List[GraphPassInstrument] = []
        self.configs: Dict[str, Any] = {
            # target precision:
            # [None, 'float16', 'bfloat16', 'float32']
            'precision': None,

            # target reduce precision:
            # [None, 'float16', 'float32']
            'reduce_precision': None,

            # mma primitive:
            # ['simt', 'wmma', 'mma']
            'mma': 'simt',

            # parallel k
            # ['default', 'disabled', 2, 4, ...]
            'parallel_k': 'default',

            # print lower details
            'verbose': False
        }

    def __enter__(self) -> PassContext:
        self._stack.append(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        popped = self._stack.pop()
        assert popped == self

    @classmethod
    def current(cls):
        """
        Get the current pass context.

        Returns
        -------
        ret: PassContext
            The current pass context.
        """
        if len(cls._stack) == 0:
            cls._stack.append(PassContext())
        return cls._stack[-1]

    def set_precision(self, dtype: Optional[str] = None) -> PassContext:
        """
        Set the target precision to use as the output of most operators. To retain the accuracy,
        some operators will still use the original data type.

        Parameters
        ----------
        dtype: Optional[str]
            The target dtype to mix the precision of the model. Candidates:
            - None
              Do not mix the precision.
            - 'float16'
              Convert the model into float16 data type.
            - 'bfloat16'
              Convert the model into bfloat16 data type.
            - 'float32'
              Convert the model into float32 data type.
        """
        self.configs['precision'] = dtype
        return self

    def set_reduce_precision(self, dtype: Optional[str] = None) -> PassContext:
        """
        Set the target precision used for accumulation results. Operators like reduce_mean, reduce_avg,
        matrix multiplication and convolution will reduce along some dimensions. We might want to use a
        data type with more precision to accumulate the results for more accuracy.

        Parameters
        ----------
        dtype: Optional[str]
        The target dtype to use for accumulation.
            - None
              Use the same as inputs of operators.
            - 'float16'
              Use 'float16' to accumulate. Only valid when set_precision('float16') has been used.
            - 'float32'
              Use 'float32' to accumulate.
        """
        self.configs['reduce_precision'] = dtype
        return self

    def set_verbose(self) -> PassContext:
        """
        Allow each graph level passes to print detailed information related to its lowering and optimization.
        """
        self.configs['verbose'] = True
        return self

    def set_mma(self, mma: str) -> PassContext:
        """
        Specify the matrix-multiply-accumulate (mma) computation primitives used in matrix multiplication and
        convolution.
        Parameters
        ----------
        mma: str
            The mma computation primitive to use. Candidates:
            - 'simt'
               Use cuda cores.
            - 'wmma'
               Use wmma instructions.
            - 'mma'
               Use mma instructions.
        """
        self.configs['mma'] = mma
        return self

    def set_parallel_k(self, disabled=False, default=False, search=False, nparts: Optional[int] = None):
        """
        Set the strategy to parallel on reduction dimension for matrix multiplication and convolution.

        Only one of the three parameters should be specified.

        Parameters
        ----------
        disabled: bool
            Disable the parallelization on reduction dimension.

        default: bool
            Allow hidet to figure our the parallel factor.

        search: bool
            Whether to search the k.

        nparts: Optional[int]
            Use a fixed factor.
        """
        if sum([disabled, default, search, nparts is not None]) > 1:
            raise ValueError('Only one of parameters should be set.')
        if disabled:
            self.configs['parallel_k'] = 'disabled'
        if default:
            self.configs['parallel_k'] = 'default'
        if search:
            self.configs['parallel_k'] = 'search'
        if nparts is not None:
            self.configs['parallel_k'] = nparts

    def save_graph_instrument(self, out_dir) -> PassContext:
        """
        Save the computation graph after each pass to given output directory.

        Parameters
        ----------
        out_dir: str
            The directory to save graph.
        """
        from .instruments.save_graph_instrument import SaveGraphInstrument
        self.instruments.append(SaveGraphInstrument(out_dir))
        return self

    def profile_pass_instrument(self, log_file: Optional[str] = None, print_stdout: bool = False) -> PassContext:
        """
        Profile the time of each pass.

        Parameters
        ----------
        log_file: Optional[str]
            When given, write the elapsed time for each pass to this file.

        print_stdout: bool
            Whether to print the elapsed time for each pass to standard output.
        """
        from .instruments.profile_instrument import ProfileInstrument
        self.instruments.append(ProfileInstrument(log_file, print_stdout))
        return self


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
