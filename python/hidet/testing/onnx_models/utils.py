from typing import List
import tempfile
import os
import torch
import onnx
import hidet
from torch import nn


def export_torch_to_onnx(
        onnx_path: str,
        model: nn.Module,
        input_names: List[str],
        inputs: List[torch.Tensor],
        nocache=False
):
    # onnx_path = hidet_cache_file('onnx', 'bert', f'{name}.onnx')
    if nocache and os.path.exists(onnx_path):
        os.remove(onnx_path)
    if not os.path.exists(onnx_path):
        model.eval()
        model(*inputs)
        _, path = tempfile.mkstemp()
        torch.onnx.export(model,
                          args=tuple(inputs),
                          f=path,
                          training=torch.onnx.TrainingMode.PRESERVE,
                          input_names=input_names,
                          opset_version=12,
                          do_constant_folding=True)
        dirname = os.path.dirname(onnx_path)
        os.makedirs(dirname, exist_ok=True)
        os.rename(path, onnx_path)
        onnx.checker.check_model(onnx_path)

    hidet_inputs = [hidet.array(torch_tensor.numpy()).cuda() for torch_tensor in inputs]
    return onnx_path, input_names, hidet_inputs
