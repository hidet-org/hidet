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
from hidet.graph.ir.flow_graph import FlowGraph


class GraphPassInstrument:
    """Graph pass instrument.

    This class defines the interface for graph pass instruments. An instrument
    defines the functions that will be called before and after each pass. This can
    be used to collect the information of graph passes. Currently, the instrument
    does not support modifying the flow graph passed to it (such functionality should
    be implemented as graph pass).

    To define a custom graph pass instrument and use it:

    .. code-block:: python

        import hidet

        # define custom instrument and implement instrument functions
        class MyInstrument(hidet.graph.GraphPassInstrument):
            def before_all_passes(self, graph: FlowGraph) -> None:
                print('before all passes')

            def before_pass(self, pass_name: str, graph: FlowGraph) -> None:
                print('before pass', pass_name)

            def after_pass(self, pass_name: str, graph: FlowGraph) -> None:
                print('after pass', pass_name)

            def after_all_passes(self, graph: FlowGraph) -> None:
                print('after all passes')

        graph = hidet.graph.FlowGraph(outputs=[])   # empty flow graph
        with hidet.graph.PassContext() as ctx:
            # add custom instrument to pass context
            ctx.instruments.append(MyInstrument())
            # optimize flow graph
            hidet.graph.optimize(graph)

    We can get output like

    .. code-block:: text

        before all passes
        before pass FoldConstantPass
        after pass FoldConstantPass
        before pass PatternTransformPass
        after pass PatternTransformPass
        ...
        after all passes

    """

    def before_all_passes(self, graph: FlowGraph) -> None:
        """Called before process all passes.

        Parameters
        ----------
        graph: FlowGraph
            The flow graph before applying all passes.
        """

    def before_pass(self, pass_name: str, graph: FlowGraph) -> None:
        """Called before each pass.

        Parameters
        ----------
        pass_name: str
            The name of the pass that is going to be applied.

        graph: FlowGraph
            The flow graph before applying the pass.
        """

    def after_pass(self, pass_name: str, graph: FlowGraph) -> None:
        """Called after each pass.

        Parameters
        ----------
        pass_name: str
            The name of the pass that has been applied.

        graph: FlowGraph
            The flow graph after applied the pass.
        """

    def after_all_passes(self, graph: FlowGraph) -> None:
        """Called after applying all passes.

        Parameters
        ----------
        graph: FlowGraph
            The flow graph after applying all passes.
        """
