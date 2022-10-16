from typing import List
import pytest
import numpy as np
import os
import onnx
import onnxruntime
import hidet.utils
from hidet import symbol_like, Tensor
from hidet.testing.onnx_models import get_onnx_model


def check_model(model_path: str, input_names: List[str], input_tensors: List[Tensor], mode: str):
    onnx.checker.check_model(model_path)

    # onnx
    onnx_session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])  # use cpu executor for high accuracy
    onnx_outputs = onnx_session.run(None, input_feed={name: tensor.numpy() for name, tensor in zip(input_names, input_tensors)})

    # hidet
    hidet_model = hidet.graph.frontend.from_onnx(model_path)
    hidet_inputs = [hidet.array(tensor).cuda() for tensor in input_tensors]

    if mode == 'imperative':
        hidet_outputs = hidet_model(*hidet_inputs)
    elif mode == 'traced' or mode == 'opt':
        symbol_inputs = [symbol_like(tensor) for tensor in hidet_inputs]
        symbol_outputs = hidet_model(*symbol_inputs)
        graph = hidet.trace_from(symbol_outputs, symbol_inputs)
        if mode == 'opt':
            model_name = os.path.splitext(os.path.basename(model_path))[0]
            out_dir = os.path.join('./outs/', model_name)
            with hidet.graph.PassContext() as ctx:
                ctx.save_graph_instrument(out_dir)
                graph = hidet.graph.optimize(graph)
        hidet_outputs = graph(*hidet_inputs)
    else:
        raise ValueError()

    if isinstance(hidet_outputs, Tensor):
        hidet_outputs = [hidet_outputs]
    hidet_outputs = [tensor.numpy() for tensor in hidet_outputs]

    assert len(onnx_outputs) == len(hidet_outputs)
    for onnx_output, hidet_output in zip(onnx_outputs, hidet_outputs):
        np.testing.assert_allclose(actual=hidet_output, desired=onnx_output, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    "model_name",
    [
        # 'resnet50',
        # 'inception_v3',
        # 'mobilenet_v2',
        'bert',
        # 'gpt2'
    ]
)
@pytest.mark.parametrize(
    "batch_size",
    [1]
)
@pytest.mark.parametrize(
    "mode",
    [
        # 'traced',
        # 'imperative',
        'opt'
    ]
)
def test_onnx_model(model_name: str, batch_size: int, mode: str):
    assert model_name in ['resnet50', 'inception_v3', 'mobilenet_v2', 'bert', 'bart', 'gpt2']
    assert mode in ['imperative', 'traced', 'opt']

    print('checking model {} in {} mode'.format(model_name, mode))
    model_path, input_names, input_tensors = get_onnx_model(model_name, batch_size=batch_size)
    check_model(model_path, input_names, input_tensors, mode)


if __name__ == '__main__':
    pytest.main([__file__])
