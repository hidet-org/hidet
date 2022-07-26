from typing import List, Optional
import time
from hidet.tos import randn_like, zeros_like
from hidet.ffi import cuda
from hidet.runtime.storage import CudaMemoryPool
from hidet.runtime.cuda_stream import CudaStream
from hidet.tos import Tensor, FlowGraph


def dummy_input_like(tensor: Tensor) -> Tensor:
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
    def __init__(self, flow_graph: FlowGraph):
        flow_graph.update_nodes()
        self.flow_graph = flow_graph
        self.mem_pool = CudaMemoryPool(
            block_size=4096,
            max_reserve_size=10 * 1024 ** 3
        )
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
        return self.inputs

    def get_output_tensors(self) -> List[Tensor]:
        return self.outputs

    def set_input_tensors(self, input_tensors: List[Tensor]):
        if len(input_tensors) != len(self.inputs):
            raise ValueError('Expect {} input tensors, got {}'.format(len(self.inputs), len(input_tensors)))
        for idx, tensor in enumerate(input_tensors):
            self.set_input_tensor(idx, tensor)

    def set_input_tensor(self, idx: int, input_tensor: Tensor):
        src = input_tensor
        dst = self.inputs[idx]
        if src.device != 'cuda':
            src = src.cuda()
        if src.dtype != dst.dtype:
            msg = 'The i-th {} input tensor expect data type {}, but got a tensor with data type {}.'.format(idx, dst.dtype, src.dtype)
            raise ValueError(msg)
        if any(a != b for a, b in zip(input_tensor.shape, self.inputs[idx].shape)):
            msg = 'The i-th {} input tensor expect shape {}, bot got a tensor with shape {}'.format(idx, dst.shape, src.shape)
            raise ValueError(msg)
        cuda.memcpy_async(src.storage.addr, dst.storage.addr, num_bytes=dst.nbytes, kind=cuda.DeviceToDevice)

    def run_with_inputs(self, inputs: List[Tensor], stream: Optional[CudaStream] = None) -> List[Tensor]:
        self.set_input_tensors(inputs)
        cuda.device_synchronize()
        self.run(stream)
        cuda.device_synchronize()
        return self.get_output_tensors()

    def run(self, stream: Optional[CudaStream] = None):
        self.cuda_graph_exec.launch(stream)

    def profile(self, warmup, number, repeat) -> List[float]:
        latency_list = []
        for i in range(warmup):
            self.run()
        for i in range(repeat):
            cuda.device_synchronize()
            start = time.time()
            for j in range(number):
                self.run()
            cuda.device_synchronize()
            end = time.time()
            latency_list.append((end - start) / number)
        return latency_list

    def __del__(self):
        self.mem_pool.storage_device.freeze(False)


def create_cuda_graph(flow_graph: FlowGraph) -> CudaGraph:
    exec_ctx = CudaGraph(flow_graph)
    return exec_ctx
