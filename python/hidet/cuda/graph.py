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
# pylint: disable=no-name-in-module, c-extension-no-member
from typing import List, Sequence, Optional
from cuda import cudart
from cuda.cudart import cudaGraphExec_t
from hidet.graph.tensor import Tensor, zeros_like, randn_like
from hidet.runtime.storage import MemoryPool, CudaMemoryAPI, memory_pool
from hidet.runtime.device import Device
from hidet.utils import same_list, exiting
from .device import current_device
from .stream import Stream, StreamContext, current_stream
from .memory import memcpy_async


class FreezableMemoryAPI(CudaMemoryAPI):
    def __init__(self, device: Device):
        super().__init__(device)
        self.frozen: bool = False

    def malloc(self, nbytes: int) -> int:
        if self.frozen:
            raise RuntimeError('Cannot malloc when the memory device is frozen')
        return super().malloc(nbytes)

    def free(self, addr: int):
        if self.frozen:
            raise RuntimeError('Cannot free when the memory device is frozen')
        super().free(addr)

    def freeze(self):
        self.frozen = True

    def unfreeze(self):
        self.frozen = False


class CudaGraphCapture:
    def __init__(self):
        self.stream = Stream()
        self.stream_context = StreamContext(self.stream)
        self.captured_graph: Optional[cudart.cudaGraph_t] = None

    def __enter__(self):
        self.stream_context.__enter__()
        (err,) = cudart.cudaStreamBeginCapture(
            self.stream.handle(), cudart.cudaStreamCaptureMode.cudaStreamCaptureModeGlobal
        )
        if err != 0:
            raise RuntimeError("cudaStreamBeginCapture failed with error: {}".format(err.name))

    def __exit__(self, exc_type, exc_val, exc_tb):
        err, self.captured_graph = cudart.cudaStreamEndCapture(self.stream.handle())
        assert err == 0, err
        self.stream_context.__exit__(exc_type, exc_val, exc_tb)

    def __del__(self, is_shutting_down=exiting.is_exiting):
        if is_shutting_down():
            return
        if self.captured_graph is not None:
            (err,) = cudart.cudaGraphDestroy(self.captured_graph)
            assert err == 0, err

    def instantiate(self) -> cudaGraphExec_t:
        if self.captured_graph is None:
            raise RuntimeError("cuda graph has not been captured yet")
        err, graph_exec = cudart.cudaGraphInstantiateWithFlags(self.captured_graph, 0)
        if err != 0:
            raise RuntimeError("cudaGraphInstantiate failed with error: {}".format(err.name))
        return graph_exec


class CudaGraph:
    """
    A CUDA graph that executes a :class:`~hidet.graph.ir.flow_graph.FlowGraph` on the GPU.

    You can create the CUDA graph by calling :meth:`~hidet.graph.ir.flow_graph.FlowGraph.cuda_graph`.

    Parameters
    ----------
    flow_graph: FlowGraph
        The flow graph to be executed.
    """

    def __init__(self, flow_graph):
        from hidet.graph.ir.flow_graph import FlowGraph

        flow_graph: FlowGraph

        self._memory_api: FreezableMemoryAPI = FreezableMemoryAPI(Device('cuda', current_device()))
        self._memory_pool: MemoryPool = MemoryPool(
            memory_api=self._memory_api, block_size=4096, max_reserve_size=10 * 1024**3
        )
        self._graph_capture: CudaGraphCapture = CudaGraphCapture()
        self._flow_graph: FlowGraph = flow_graph
        self._inputs: List[Tensor]
        self._outputs: List[Tensor]

        with memory_pool(self._memory_pool):
            # update the nodes and inputs of the flow graph
            flow_graph.update_nodes()

            # prepare the dummpy inputs
            inputs = []
            for tensor in flow_graph.inputs:
                if tensor.is_symbolic():
                    if tensor.dtype.is_float():
                        inputs.append(randn_like(tensor, device='cuda'))
                    else:
                        inputs.append(zeros_like(tensor, device='cuda'))
                else:
                    inputs.append(tensor)
            self._inputs = inputs

            # run and capture the graph execution
            flow_graph.forward(*self._inputs)  # warm up, avoid memory allocation during capturing
            flow_graph.forward(*self._inputs)
            self._memory_api.freeze()
            with self._graph_capture:
                outputs = flow_graph.forward(*self._inputs)

            # process the outputs
            self._outputs = [outputs] if isinstance(outputs, Tensor) else outputs

        # instantiate the captured graph
        self._graph_exec: cudaGraphExec_t = self._graph_capture.instantiate()

    def __call__(self, *inputs: Tensor):
        if len(inputs) == 0:
            return self.run()
        else:
            return self.run(inputs)

    def __del__(self, is_shutting_down=exiting.is_exiting):
        if is_shutting_down():
            return
        if hasattr(self, '_graph_exec'):
            (err,) = cudart.cudaGraphExecDestroy(self._graph_exec)
            assert err == 0, err
            self._memory_api.unfreeze()

    def copy_inputs(self, inputs: Sequence[Tensor], stream: Optional[Stream]):
        if len(inputs) != len(self._inputs):
            raise ValueError("the number of inputs does not match")
        for src, dst in zip(inputs, self._inputs):
            if src.is_symbolic():
                raise ValueError("the input tensor is symbolic")
            if not same_list(src.shape, dst.shape):
                raise ValueError("the shape of input does not match")
            if src.dtype != dst.dtype:
                raise ValueError("the dtype of input does not match")
            if src.device != dst.device:
                raise ValueError("the device of input does not match")
            memcpy_async(dst=dst.storage.addr, src=src.storage.addr, num_bytes=dst.nbytes, stream=stream)

    @property
    def inputs(self) -> List[Tensor]:
        """
        The inputs of the cuda graph.
        """
        return self._inputs

    @property
    def outputs(self) -> List[Tensor]:
        """
        The outputs of the cuda graph.
        """
        return self._outputs

    def run(self, inputs: Optional[Sequence[Tensor]] = None) -> List[Tensor]:
        """
        Run the cuda graph synchronously. If the inputs are provided, the inputs will be copied to the
        internal inputs of the cuda graph before running.

        Parameters
        ----------
        inputs: Optional[Sequence[Tensor]]
            The optional inputs to run the cuda graph.

        Returns
        -------
        outputs: List[Tensor]
            The outputs of the cuda graph.
        """
        stream = current_stream()
        self.run_async(inputs, stream)
        stream.synchronize()
        return self.outputs

    def run_async(self, inputs: Optional[Sequence[Tensor]] = None, stream: Optional[Stream] = None) -> List[Tensor]:
        """
        Run the cuda graph asynchronously. If the inputs are provided, the inputs will be copied to the
        internal inputs of the cuda graph before running.

        Parameters
        ----------
        inputs: Optional[Sequence[Tensor]]
            The optional inputs to run the cuda graph.

        stream: Optional[Stream]
            The optional stream to run the cuda graph. If not provided, the current stream will be used.

        Returns
        -------
        outputs: List[Tensor]
            The outputs of the cuda graph.
        """
        if stream is None:
            stream = current_stream()
        if inputs is not None:
            self.copy_inputs(inputs, stream)
        cudart.cudaGraphLaunch(self._graph_exec, stream)
        return self.outputs
