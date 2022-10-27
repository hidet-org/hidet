from typing import List, Optional, Union
from hidet.ffi import cuda
from hidet.runtime.storage import CudaMemoryPool
from hidet.runtime.cuda_stream import CudaStream
from hidet.graph.tensor import Tensor
from hidet.graph.ir.flow_graph import FlowGraph
from hidet.testing import benchmark_func


def dummy_input_like(tensor: Tensor) -> Tensor:
    from hidet.graph import randn_like, zeros_like

    if tensor.dtype in ['float32', 'float16']:
        return randn_like(tensor)
    elif tensor.dtype in ['int64', 'int32', 'int8', 'uint64', 'uint32', 'uint8']:
        return zeros_like(tensor)
    else:
        raise ValueError('Can not generate dummy input for data type {}'.format(tensor.dtype))


class CudaGraphExec:
    def __init__(self, exec_handle: int):
        self.exec_handle = exec_handle

    def __del__(self):
        cuda.destroy_graph_exec(self.exec_handle)

    def launch(self, stream: Optional[CudaStream] = None):
        stream_handle = stream.handle if stream else 0
        cuda.launch_graph_exec(self.exec_handle, stream_handle)


class CudaGraphImpl:
    def __init__(self):
        self.stream = CudaStream()
        self.graph_handle: Optional[int] = None

    def __enter__(self):
        self.stream.__enter__()
        cuda.stream_begin_capture(self.stream.handle)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stream.__exit__(exc_type, exc_val, exc_tb)
        if self.graph_handle is not None:
            cuda.destroy_graph(self.graph_handle)
        self.graph_handle = cuda.stream_end_capture(self.stream.handle)

    def __del__(self):
        if self.graph_handle is not None:
            cuda.destroy_graph(self.graph_handle)

    def instantiate(self) -> CudaGraphExec:
        exec_handle = cuda.instantiate_graph(self.graph_handle)
        graph_exec = CudaGraphExec(exec_handle)
        return graph_exec


class CudaGraph:
    """CUDA Graph wrapper class.

    The cuda graph tracks the kernel launches on the cuda device and replays them efficiently. To use the cuda graph,
    we should

    1. Use function :func:`FlowGraph.cuda_graph() <hidet.graph.FlowGraph.cuda_graph>` to create a cuda graph from an
       existing flow graph.
    2. Use :func:`~hidet.runtime.cuda_graph.CudaGraph.set_input_tensors` or
       :func:`~hidet.runtime.cuda_graph.CudaGraph.set_input_tensor` to set the values of the cuda graph input tensors.
       These functions would copy the contents of the given tensors to the input tensors of the cuda graph.
    3. Run the cuda graph with :func:`~hidet.runtime.cuda_graph.CudaGraph.run`.
    4. Access the results through the cuda graph output tensors :attr:`~hidet.runtime.cuda_graph.CudaGraph.outputs`.

    Attributes
    ----------
    flow_graph: FlowGraph
        The flow graph that this cuda graph is created from.

    inputs: List[Tensor]
        The input tensors of the cuda graph.

    outputs: List[Tensor]
        The output tensors of the cuda graph.

    mem_pool: CudaMemoryPool
        The memory pool used by this cuda graph.
    """

    def __init__(self, flow_graph: FlowGraph):
        flow_graph.update_nodes()
        self.flow_graph = flow_graph
        self.mem_pool = CudaMemoryPool(block_size=4096, max_reserve_size=10 * 1024**3)
        self.cuda_graph_impl = CudaGraphImpl()
        with self.mem_pool:
            self.inputs = [dummy_input_like(tensor) for tensor in flow_graph.inputs]
            # run twice to avoid any memory allocation during capturing
            self.outputs = flow_graph.forward(*self.inputs)
            self.outputs = flow_graph.forward(*self.inputs)
            self.mem_pool.storage_device.freeze(True)
            with self.cuda_graph_impl:
                self.outputs = flow_graph.forward(*self.inputs)
            if isinstance(self.outputs, Tensor):
                self.outputs = [self.outputs]
            # self.mem_pool.storage_device.freeze(False)
        self.cuda_graph_exec = self.cuda_graph_impl.instantiate()

    def get_input_tensors(self) -> List[Tensor]:
        """Get the tensors to store the inputs of the cuda graph.

        Returns
        -------
        ret: List[Tensor]
            The input tensors.
        """
        return self.inputs

    def get_output_tensors(self) -> List[Tensor]:
        """Get the tensors to store the outputs of the cuda graph.

        Returns
        -------
        ret: List[Tensor]
            The output tensors.
        """
        return self.outputs

    def set_input_tensors(self, input_tensors: List[Tensor], stream: Optional[CudaStream] = None):
        """Set the values of input tensors.

        This function copies the contents of the given tensors to the input tensors of the cuda graph.

        Parameters
        ----------
        input_tensors: List[Tensor]
            The tensors stored the input contents.

        stream: Optional[CudaStream]
            The stream to copy the contents.
        """
        if len(input_tensors) != len(self.inputs):
            raise ValueError('Expect {} input tensors, got {}'.format(len(self.inputs), len(input_tensors)))
        for idx, tensor in enumerate(input_tensors):
            self.set_input_tensor(idx, tensor, stream)

    def set_input_tensor(self, idx: int, input_tensor: Tensor, stream: Optional[CudaStream] = None):
        """Set the content of the input tensor with index `idx`.

        This function copies the contents of the given tensor to the input tensor of the cuda graph.

        Parameters
        ----------
        idx: int
            The index of input tensor.

        input_tensor: Tensor
            The tensor that contains the content.

        stream: Optional[CudaStream]
            The stream to copy the contents.
        """
        src = input_tensor
        dst = self.inputs[idx]
        if src.device != 'cuda':
            src = src.cuda()
        if src.dtype != dst.dtype:
            msg = 'The i-th {} input tensor expect data type {}, but got a tensor with data type {}.'.format(
                idx, dst.dtype, src.dtype
            )
            raise ValueError(msg)
        if any(a != b for a, b in zip(input_tensor.shape, self.inputs[idx].shape)):
            msg = 'The i-th {} input tensor expect shape {}, bot got a tensor with shape {}'.format(
                idx, dst.shape, src.shape
            )
            raise ValueError(msg)
        cuda.memcpy_async(
            src.storage.addr,
            dst.storage.addr,
            num_bytes=dst.nbytes,
            kind=cuda.DeviceToDevice,
            stream=stream.handle if stream else 0,
        )

    def run_with_inputs(self, inputs) -> List[Tensor]:
        """Run the cuda graph with given inputs.

        Parameters
        ----------
        inputs: List[Tensor]
            The input tensors.

        stream: Optional[CudaStream]
            The cuda stream to launch the cuda graph on.

        Returns
        -------
        ret: List[Tensor]
            The output tensors.
        """
        self.set_input_tensors(inputs)
        cuda.device_synchronize()
        self.run()
        cuda.device_synchronize()
        return self.get_output_tensors()

    def run(self, stream: Optional[CudaStream] = None):
        """Run the cuda graph.

        Before run this function, :func:`set_input_tensors` or :func:`set_input_tensor` should be used set the input
        tensor contents.

        Access attribute :attr:`outputs` to get the output contents.

        Parameters
        ----------
        stream: Optional[CudaStream]
            The cuda stream to run the cuda graph. None indicates default stream.
        """
        self.cuda_graph_exec.launch(stream)

    def profile(self, warmup, number, repeat, median=True) -> Union[float, List[float]]:
        return benchmark_func(lambda: self.run(), warmup, number, repeat, median)

    def __del__(self):
        self.mem_pool.storage_device.freeze(False)


def create_cuda_graph(flow_graph: FlowGraph) -> CudaGraph:
    exec_ctx = CudaGraph(flow_graph)
    return exec_ctx
