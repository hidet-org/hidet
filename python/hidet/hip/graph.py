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
from typing import List, Sequence, Optional, Any, Callable

try:
    from hip import hip
    from hip.hip import hipGraphExec_t
except ImportError as e:
    raise RuntimeError(
        "Please install hip-python with \
            `python3 -m pip install -i https://test.pypi.org/simple hip-python>=5.7.1` and try again"
    ) from e
from hidet.graph.tensor import Tensor
from hidet.runtime.storage import MemoryPool, HipMemoryAPI, memory_pool
from hidet.runtime.device import Device
from hidet.utils import same_list, exiting
from .device import current_device
from .stream import Stream, StreamContext, current_stream
from .memory import memcpy_async


class HipGraphCreationError(Exception):
    pass


class FreezableMemoryAPI(HipMemoryAPI):
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


class HipGraphCapture:
    def __init__(self):
        self.stream = Stream()
        self.stream_context = StreamContext(self.stream)
        self.captured_graph: Optional[hip.hipGraph_t] = None

    def __enter__(self):
        self.stream_context.__enter__()
        (err,) = hip.hipStreamBeginCapture(self.stream.handle(), hip.hipStreamCaptureMode.hipStreamCaptureModeGlobal)
        if err != 0:
            raise RuntimeError("hipStreamBeginCapture failed with error: {}".format(str(err)))

    def __exit__(self, exc_type, exc_val, exc_tb):
        err, self.captured_graph = hip.hipStreamEndCapture(self.stream.handle())
        assert err == 0, err
        self.stream_context.__exit__(exc_type, exc_val, exc_tb)

    def __del__(self, is_shutting_down=exiting.is_exiting):
        if is_shutting_down():
            return
        if self.captured_graph is not None:
            (err,) = hip.hipGraphDestroy(self.captured_graph)
            assert err == 0, err

    def instantiate(self) -> hipGraphExec_t:
        if self.captured_graph is None:
            raise RuntimeError("hip graph has not been captured yet")
        err, graph_exec = hip.hipGraphInstantiateWithFlags(self.captured_graph, 0)
        if err != 0:
            raise RuntimeError("hipGraphInstantiate failed with error: {}".format(str(err)))
        return graph_exec


class HipGraph:
    """
    Create a hip graph to capture and replay the execution of a series of hip kernels launched in a function.

    The graph is created by calling the constructor with the following arguments:

    Parameters
    ----------
    f_create_inputs: Callable[[], List[Tensor]]
        A function that creates the input tensors of the graph. This function is called before f_run.
    f_run: Callable[[List[Tensor]], List[Tensor]]
        A function that runs the graph. Only the hip kernels launched in this function will be captured. Rerunning
        this function must launch the same hip kernels in the same order. The input tensors of this function will be
        the output tensors of the f_create_inputs function.
    ref_objs: Any
        The objects that should keep alive during the lifetime of the hip graph. It may contain the weight tensors
        that are used in the graph.
    """

    def __init__(
        self,
        f_create_inputs: Callable[[], List[Tensor]],
        f_run: Callable[[List[Tensor]], List[Tensor]],
        ref_objs: List[Any],
    ):
        self._memory_api: FreezableMemoryAPI = FreezableMemoryAPI(Device('hip', current_device()))
        self._memory_pool: MemoryPool = MemoryPool(
            memory_api=self._memory_api, block_size=4096, max_reserve_size=10 * 1024**3
        )
        self._graph_capture: HipGraphCapture = HipGraphCapture()
        self._inputs: List[Tensor] = []
        self._outputs: List[Tensor] = []
        self._ref_objs: List[Any] = ref_objs

        with memory_pool(self._memory_pool):
            # create the input tensors
            self._inputs = f_create_inputs()

            # warmup the run function
            num_warmup = 2
            for _ in range(num_warmup):
                f_run(self._inputs)

            # capture the hip graph
            self._memory_api.freeze()
            with self._graph_capture:
                self._outputs = f_run(self._inputs)

        # instantiate the hip graph
        self._graph_exec: hipGraphExec_t = self._graph_capture.instantiate()

    def __call__(self, *inputs: Tensor):
        if len(inputs) == 0:
            self.run()
        else:
            self.run(inputs)
        if len(self.outputs) == 1:
            return self.outputs[0]
        else:
            return self.outputs

    def __del__(self, is_shutting_down=exiting.is_exiting):
        if is_shutting_down():
            return
        if hasattr(self, '_graph_exec'):
            (err,) = hip.hipGraphExecDestroy(self._graph_exec)
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
        The inputs of the hip graph.
        """
        return self._inputs

    @property
    def outputs(self) -> List[Tensor]:
        """
        The outputs of the hip graph.
        """
        return self._outputs

    def run(self, inputs: Optional[Sequence[Tensor]] = None) -> List[Tensor]:
        """
        Run the hip graph synchronously. If the inputs are provided, the inputs will be copied to the
        internal inputs of the hip graph before running.

        Parameters
        ----------
        inputs: Optional[Sequence[Tensor]]
            The optional inputs to run the hip graph.

        Returns
        -------
        outputs: List[Tensor]
            The outputs of the hip graph.
        """
        stream = current_stream()
        self.run_async(inputs, stream)
        stream.synchronize()
        return self.outputs

    def run_async(self, inputs: Optional[Sequence[Tensor]] = None, stream: Optional[Stream] = None) -> List[Tensor]:
        """
        Run the hip graph asynchronously. If the inputs are provided, the inputs will be copied to the
        internal inputs of the hip graph before running.

        Parameters
        ----------
        inputs: Optional[Sequence[Tensor]]
            The optional inputs to run the hip graph.

        stream: Optional[Stream]
            The optional stream to run the hip graph. If not provided, the current stream will be used.

        Returns
        -------
        outputs: List[Tensor]
            The outputs of the hip graph.
        """
        if stream is None:
            stream = current_stream()
        if inputs is not None:
            self.copy_inputs(inputs, stream)
        (err,) = hip.hipGraphLaunch(self._graph_exec, int(stream))
        assert err == 0, err
        return self.outputs
