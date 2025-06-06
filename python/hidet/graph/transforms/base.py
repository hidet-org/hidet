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
from __future__ import annotations
from typing import List, Optional, Dict, Any
import logging

import hidet.option
from hidet.graph.flow_graph import FlowGraph
from hidet.graph.transforms.graph_patterns.base import SubgraphRewriteRule
from .instruments import GraphPassInstrument

logger = logging.Logger(name='hidet.graph.transforms', level=logging.INFO)
logger.addHandler(logging.StreamHandler())


class PassContext:
    """Graph-level pass context.

    Use the pass context to control the behavior of optimization passes. Normally, we can
    optimize a flow graph by directly calling :func:`hidet.graph.optimize`:

    .. code-block:: python

      graph_opt = hidet.graph.optimize(graph)

    This will optimize the given flow graph in a default context.

    To customize the optimizations, run the :func:`~hidet.graph.optimize` function with in
    a custom :class:`hidet.graph.PassContext`:

    .. code-block:: python

      with hidet.graph.PassContext() as ctx:
          # config the contexts
          ctx.profile_pass_instrument(print_stdout=True)  # print elapsed time for each pass
          ctx.save_graph_instrument(out_dir='./outs')  # save the output of each pass as text
          ctx.set_precision(dtype='float16')  # use float16 as the data type
          ctx.set_reduce_precision(dtype='float32')  # use float32 for reduction accumulation
          ctx.set_mma('mma')  # use TensorCore in NVIDIA GPUs to accelerate matmul and conv2d
          ...   # other configs

          # call optimize function
          graph_opt = hidet.graph.optimize(graph)

    Please refer to the member functions of this class for the available configs and their usage.

    Attributes
    ----------
    instruments: List[GraphPassInstrument]
        The graph pass instruments that will be applied before and after each pass. The instruments
        will be applied in order. See :class:`hidet.graph.GraphPassInstrument` on how to add custom
        instrument.

    configs: Dict[str, Any]
        The current configs of the pass context.
    """

    _stack: List['PassContext'] = []

    def __init__(self):
        self.instruments: List[GraphPassInstrument] = []
        self.configs: Dict[str, Any] = {
            # target precision:
            # [None, 'int8', 'float16', 'bfloat16', 'float32']
            'precision': None,
            # selectively quantize the given graph patterns
            'quantize_patterns': [],
            # target reduce precision:
            # [None, 'float16', 'float32']
            'reduce_precision': None,
            # mma primitive:
            # ['simt', 'mma']
            'mma': 'mma',
            # print lower details
            'verbose': False,
            # Allow source graph removal.
            # It is used to get rid of unnecessary tensors during the optimizations
            'allow_source_graph_removal': False,
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
            - 'int8'
                Converts the model into float16 data type, then selectively quantize subgraphs
                using default quantize_patterns.
                For greater flexibility and control of quantization, use self.add_quantize_pattern(),
                to selectively quantize subgraphs using custom quantize_patterns.
            - 'float16'
              Convert the model into float16 data type.
            - 'bfloat16'
              Convert the model into bfloat16 data type.
            - 'float32'
              Convert the model into float32 data type.
        """
        if dtype == 'int8':
            self.add_quantize_rules(hidet.graph.quant.default_patterns())
            self.configs['precision'] = 'float16'
        else:
            self.configs['precision'] = dtype
        return self

    def add_quantize_rules(self, patterns: List[SubgraphRewriteRule]) -> PassContext:
        """
        Adds selective quantization rules to the pass context.

        Parameters
        ----------
        pattern: Optional[List[SubgraphRewriteRule]]
            The pattern to selectively quantize.

            - List[SubgraphRewriteRule]
              Adds new rules on top of what is already there. The new rules will be applied
              after the existing ones.
        """

        if patterns is not None:
            for pat in patterns:
                if isinstance(pat, SubgraphRewriteRule):
                    self.configs['quantize_patterns'].append(pat)
                elif issubclass(pat, SubgraphRewriteRule):
                    self.configs['quantize_patterns'].append(pat())
        else:
            self.configs['quantize_patterns'] = []
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
            - 'mma'
               Use mma instructions.
        """
        self.configs['mma'] = mma
        return self

    def save_graph_instrument(self, out_dir) -> PassContext:
        """
        Save the computation graph after each pass to given output directory.

        Parameters
        ----------
        out_dir: str
            The directory to save graph.
        """
        from .instruments.save_graph_instrument import SaveGraphInstrument  # pylint: disable=import-outside-toplevel

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
        from .instruments.profile_instrument import ProfileInstrument  # pylint: disable=import-outside-toplevel

        self.instruments.append(ProfileInstrument(log_file, print_stdout))
        return self

    def reduce_cuda_compile_mem(self, enable: Optional[bool] = None):
        """
        Reduce CUDA memory used during compilation by using vcuda tensors, might incur compile time cost

        Parameters
        ----------
        enable: Optional[bool]
            When given, will always enable or disable this instrument.
            If no argument is given, the compiler will decide to enable this with some heuristics
        """
        from .instruments import ConvertGraphToVGPU  # pylint: disable=import-outside-toplevel

        self.instruments.append(ConvertGraphToVGPU(enable, target='cuda'))

    def reduce_hip_compile_mem(self, enable: Optional[bool] = None):
        """
        Reduce HIP memory used during compilation by using vhip tensors, might incur compile time cost

        Parameters
        ----------
        enable: Optional[bool]
            When given, will always enable or disable this instrument.
            If no argument is given, the compiler will decide to enable this with some heuristics
        """
        from .instruments import ConvertGraphToVGPU  # pylint: disable=import-outside-toplevel

        self.instruments.append(ConvertGraphToVGPU(enable, target='hip'))

    def allow_source_graph_removal(self, allow: Optional[bool] = True):
        self.configs['allow_source_graph_removal'] = allow

    def is_source_graph_removal_allowed(self):
        return self.configs['allow_source_graph_removal']


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
