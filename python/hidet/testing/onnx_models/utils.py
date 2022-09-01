from typing import List, Optional
import tempfile
import shutil
import os
import onnx
import hidet

try:
    import torch
    from torch import nn
except ImportError:
    pass


def export_torch_to_onnx(
        onnx_path: str,
        model: nn.Module,
        input_names: List[str],
        inputs: List[torch.Tensor],
        precision: Optional[str] = None,
        nocache=False
):
    # onnx_path = hidet_cache_file('onnx', 'bert', f'{name}.onnx')
    if nocache and os.path.exists(onnx_path):
        os.remove(onnx_path)
    precision_dict = {
        'float32': torch.float32,
        'float16': torch.float16
    }
    if precision:
        inputs = [t.type(precision_dict[precision]) if torch.is_floating_point(t) else t for t in inputs]
    if not os.path.exists(onnx_path):
        model.eval()
        if precision:
            assert precision in ['float32', 'float16']
            model = model.to(dtype=precision_dict[precision])
        # model(*inputs)
        _, path = tempfile.mkstemp()
        model.cuda()
        inputs = [t.cuda() for t in inputs]
        torch.onnx.export(model,
                          args=tuple(inputs),
                          f=path,
                          training=torch.onnx.TrainingMode.PRESERVE,
                          input_names=input_names,
                          opset_version=12,
                          do_constant_folding=True
                          )
        dirname = os.path.dirname(onnx_path)
        os.makedirs(dirname, exist_ok=True)
        shutil.move(path, onnx_path)
        # os.rename(path, onnx_path)    'OSError: [Errno 18] Invalid cross-device link' when src and dst are on different filesystem.
        onnx.checker.check_model(onnx_path)

    numpy_inputs = [torch_tensor.cpu().numpy() for torch_tensor in inputs]
    hidet_inputs = [hidet.array(numpy_input).cuda() for numpy_input in numpy_inputs]
    return onnx_path, input_names, hidet_inputs
