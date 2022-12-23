# pylint: disable=no-name-in-module, c-extension-no-member
from typing import List, Sequence, Optional
from cuda import cudart
from cuda.cudart import cudaGraphExec_t
from hidet.graph.tensor import Tensor, zeros_like, randn_like
from hidet.graph.ir.flow_graph import FlowGraph
from hidet.runtime.storage import CudaMemoryPool
from hidet.utils import same_list
from .stream import Stream, current_stream
from .memory import memcpy_async, cudaMemcpyKind


class CudaGraphCapture:
    def __init__(self):
        self.stream = Stream()
        self.captured_graph: Optional[cudart.cudaGraph_t] = None

    def __enter__(self):
        err = cudart.cudaStreamBeginCapture(
            self.stream.handle(), cudart.cudaStreamCaptureMode.cudaStreamCaptureModeGlobal
        )
        if err != 0:
            raise RuntimeError("cudaStreamBeginCapture failed with error: {}".format(err.name))

    def __exit__(self, exc_type, exc_val, exc_tb):
        err, self.captured_graph = cudart.cudaStreamEndCapture(self.stream.handle())
        assert err == 0

    def __del__(self):
        if self.captured_graph is not None:
            err = cudart.cudaGraphDestroy(self.captured_graph)
            assert err == 0

    def instantiate(self) -> cudaGraphExec_t:
        if self.captured_graph is None:
            raise RuntimeError("cuda graph has not been captured yet")
        err, graph_exec = cudart.cudaGraphInstantiate(self.captured_graph, 0)
        if err != 0:
            raise RuntimeError("cudaGraphInstantiate failed with error: {}".format(err.name))
        return graph_exec


class CudaGraph:
    def __init__(self, flow_graph: FlowGraph):
        self._memory_pool: CudaMemoryPool = CudaMemoryPool(block_size=4096, max_reserve_size=10 * 1024**3)
        self._graph_capture: CudaGraphCapture = CudaGraphCapture()
        self._flow_graph: FlowGraph = flow_graph
        self._inputs: List[Tensor]
        self._outputs: List[Tensor]

        with self._memory_pool:
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
            self._memory_pool.storage_device.freeze(True)
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

    def __del__(self):
        err = cudart.cudaGraphExecDestroy(self._graph_exec)
        assert err == 0

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
            memcpy_async(
                dst.storage.addr,
                src.storage.addr,
                dst.nbytes,
                kind=cudaMemcpyKind.cudaMemcpyDeviceToDevice,
                stream=stream,
            )

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
