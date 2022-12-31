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
from typing import List
import pytest
import numpy as np
import onnx
import onnxruntime
import hidet.utils
from hidet import symbol_like, Tensor
from hidet.testing.onnx_models import get_onnx_model


def check_model(model_path: str, input_names: List[str], input_tensors: List[Tensor], mode: str, dtype: str):
    onnx.checker.check_model(model_path)

    # onnx
    onnx_session = onnxruntime.InferenceSession(
        model_path, providers=['CPUExecutionProvider']
    )  # use cpu executor for high accuracy
    onnx_outputs = onnx_session.run(
        None, input_feed={name: tensor.cpu().numpy() for name, tensor in zip(input_names, input_tensors)}
    )

    # hidet
    hidet_model = hidet.graph.frontend.from_onnx(model_path)
    hidet_inputs = [hidet.asarray(tensor).cuda() for tensor in input_tensors]

    if mode == 'imperative':
        hidet_outputs = hidet_model(*hidet_inputs)
    elif mode == 'traced' or mode == 'opt':
        symbol_inputs = [symbol_like(tensor) for tensor in hidet_inputs]
        symbol_outputs = hidet_model(*symbol_inputs)
        graph = hidet.trace_from(symbol_outputs, symbol_inputs)
        if mode == 'opt':
            with hidet.graph.PassContext() as ctx:
                ctx.set_precision(dtype)
                graph = hidet.graph.optimize(graph)
        hidet_outputs = graph(*hidet_inputs)
    else:
        raise ValueError()

    if isinstance(hidet_outputs, Tensor):
        hidet_outputs = [hidet_outputs]
    hidet_outputs = [tensor.cpu().numpy() for tensor in hidet_outputs]

    assert len(onnx_outputs) == len(hidet_outputs)
    tol = {'float32': 1e-4, 'float16': 5e-2}[dtype]
    for onnx_output, hidet_output in zip(onnx_outputs, hidet_outputs):
        np.testing.assert_allclose(actual=hidet_output, desired=onnx_output, rtol=tol, atol=tol)


@pytest.mark.parametrize(
    "model_name",
    [
        'resnet50',
        # 'inception_v3',
        # 'mobilenet_v2',
        'bert',
        # 'gpt2'
    ],
)
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("dtype", ['float32', 'float16'])
@pytest.mark.parametrize("mode", ['traced', 'imperative', 'opt'])
def test_onnx_model(model_name: str, batch_size: int, dtype: str, mode: str):
    if hidet.cuda.compute_capability() < (8, 0) and dtype == 'float16':
        pytest.skip(
            'float16 will triger hidet to use fp16 tensor core (mma.m16n8k16), '
            'which is only supported on sm80 and above'
        )
    assert model_name in ['resnet50', 'inception_v3', 'mobilenet_v2', 'bert', 'bart', 'gpt2']
    assert mode in ['imperative', 'traced', 'opt']

    print('checking model {} in {} mode with dtype {}'.format(model_name, mode, dtype))
    model_path, input_names, input_tensors = get_onnx_model(model_name, batch_size=batch_size)
    check_model(model_path, input_names, input_tensors, mode, dtype)


if __name__ == '__main__':
    pytest.main([__file__])
