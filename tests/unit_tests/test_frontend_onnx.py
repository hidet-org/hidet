import pytest
import numpy as np
from hidet import randn
from hidet.tos.frontend.onnx_utils import OnnxModule
from hidet.utils import download


# noinspection PyPackageRequirements
def onnx_installed() -> bool:
    try:
        import onnx
        import onnxruntime
    except ImportError:
        return False


@pytest.mark.skipif(condition=onnx_installed(), reason='ONNX not installed')
def test_resnet50():
    import onnx
    import onnxruntime
    model_path = download(
        url='https://media.githubusercontent.com/media/onnx/models/main/vision/classification/resnet/model/resnet50-v1-7.onnx',
        file_name='onnx/resnet50-v1-7.onnx',
        progress=True)
    model = onnx.load_model(model_path)
    onnx.checker.check_model(model)
    # run use hidet
    x_hidet = randn([1, 3, 224, 224], device='cuda')
    module = OnnxModule(model)
    y_hidet = module(x_hidet)
    y_hidet = y_hidet.cpu().numpy()

    # run use onnx runtime
    onnx_infer = onnxruntime.InferenceSession(model_path)
    y_onnx = onnx_infer.run(None, {'data': x_hidet.cpu().numpy()})
    y_onnx = y_onnx[0]

    # compare
    np.testing.assert_allclose(actual=y_hidet, desired=y_onnx, rtol=1e-5, atol=1e-5)


if __name__ == '__main__':
    test_resnet50()
