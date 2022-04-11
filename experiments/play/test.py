# class Apple:
#     def __init__(self, name):
#         self.name = name
#         print('init ' + self.name)
#
#     def __del__(self):
#         print('del ' + self.name)
#
#
# class Orange:
#     def __init__(self, name):
#         self.dct = dict()
#         self.dct['a'] = []
#         self.dct['a'].append(Apple('xxx'))
#
#
#     def doit(self):
#         return self.dct['a'].pop()
#
#
# if __name__ == '__main__':
#     a = Apple('a')
#     del a
#     b = Orange('c')
#     c = b.doit()
#     del c
#     print('here')


import pytest
import numpy as np
from hidet import randn
from hidet.tos.frontend.onnx_utils import OnnxModule
from hidet.utils import download, Timer


# noinspection PyPackageRequirements
def onnx_installed() -> bool:
    try:
        import onnx
        import onnxruntime
    except ImportError:
        return False


@pytest.mark.skipif(condition=onnx_installed(), reason='ONNX not installed')
def resnet50():
    import onnx
    import onnxruntime
    # model_path = '/home/yaoyao/model_zoo/resnet50_v1.onnx'
    model_path = download('https://media.githubusercontent.com/media/onnx/models/main/vision/classification/resnet/model/resnet50-v1-7.onnx',
                          'onnx/resnet50-v1-7.onnx')
    model = onnx.load_model(model_path)
    onnx.checker.check_model(model)
    # run use hidet
    with Timer('hidet'):
        x_hidet = randn([1, 3, 224, 224], device='cuda')
        with Timer('build model'):
            module = OnnxModule(model)
        with Timer('run model'):
            y_hidet = module(x_hidet)
        y_hidet = y_hidet.cpu().numpy()

    # run use onnx runtime
    with Timer('onnx'):
        onnx_infer = onnxruntime.InferenceSession(model_path)
        y_onnx = onnx_infer.run(None, {'data': x_hidet.cpu().numpy()})
        y_onnx = y_onnx[0]

    # compare
    np.testing.assert_allclose(actual=y_hidet, desired=y_onnx, rtol=1e-5, atol=1e-5)


if __name__ == '__main__':
    resnet50()
