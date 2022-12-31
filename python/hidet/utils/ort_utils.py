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
from typing import Dict, List
import onnxruntime as ort
import hidet
from hidet import Tensor
from hidet.testing import benchmark_func


def create_ort_session(onnx_model_path, provider='CUDAExecutionProvider') -> ort.InferenceSession:
    session = ort.InferenceSession(onnx_model_path, providers=[provider])
    session.disable_fallback()
    return session


def _prepare_io_binding(session: ort.InferenceSession, inputs: Dict[str, Tensor]) -> ort.IOBinding:
    input_values: Dict[str, ort.OrtValue] = {
        name: ort.OrtValue.ortvalue_from_numpy(tensor.numpy(), device_type='cuda') for name, tensor in inputs.items()
    }
    output_names = [output.name for output in session.get_outputs()]
    io_binding = session.io_binding()
    for name, value in input_values.items():
        io_binding.bind_ortvalue_input(name, value)
    for name in output_names:
        io_binding.bind_output(name, device_type='cuda')
    return io_binding


def ort_inference(session: ort.InferenceSession, inputs: Dict[str, Tensor]) -> Dict[str, Tensor]:
    io_binding = _prepare_io_binding(session, inputs)
    session.run_with_iobinding(iobinding=io_binding)
    outputs = {
        output_node.name: hidet.asarray(value.numpy()).cuda()
        for output_node, value in zip(session.get_outputs(), io_binding.get_outputs())
    }
    return outputs


def ort_benchmark(
    session: ort.InferenceSession, dummy_inputs: Dict[str, Tensor], warmup=10, number=10, repeat=10
) -> List[float]:
    io_binding = _prepare_io_binding(session, dummy_inputs)
    return benchmark_func(
        lambda: session.run_with_iobinding(iobinding=io_binding),
        warmup=warmup,
        number=number,
        repeat=repeat,
        median=False,
    )
