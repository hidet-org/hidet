from typing import List, Optional
import tempfile
import shutil
import os
import onnx
import hidet


def export_torch_to_onnx(
    onnx_path: str, model, input_names: List[str], inputs, precision: Optional[str] = None, nocache=False
):
    """
    Export a torch model to onnx.

    Parameters
    ----------
    onnx_path: str
        Path to store the onnx file.
    model: torch.nn.Module
        The torch model to be exported.
    input_names: List[str]
        The names of the inputs in the exported onnx model.
    inputs: Sequence[torch.Tensor]
        The inputs to the model.
    precision: Optional[str]
        The precision of the exported onnx model. If None, the precision of the model is not changed.
        Candidates: 'float16', 'float32'
    nocache: bool
        If True, the onnx model will be exported even if the onnx file already exists.

    Returns
    -------
    (onnx_path, input_names, hidet_inputs): Tuple[str, List[str], List[hidet.Tensor]]
        The path to the exported onnx model, the names of the inputs in the exported onnx model, and the inputs to the
        exported onnx model.
    """

    import torch

    if nocache and os.path.exists(onnx_path):
        os.remove(onnx_path)
    precision_dict = {'float32': torch.float32, 'float16': torch.float16}
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
        torch.onnx.export(
            model,
            args=tuple(inputs),
            f=path,
            training=torch.onnx.TrainingMode.PRESERVE,
            input_names=input_names,
            opset_version=12,
            do_constant_folding=True,
        )
        dirname = os.path.dirname(onnx_path)
        os.makedirs(dirname, exist_ok=True)
        shutil.move(path, onnx_path)
        onnx.checker.check_model(onnx_path)

    numpy_inputs = [torch_tensor.cpu().numpy() for torch_tensor in inputs]
    hidet_inputs = [hidet.array(numpy_input).cuda() for numpy_input in numpy_inputs]
    return onnx_path, input_names, hidet_inputs
