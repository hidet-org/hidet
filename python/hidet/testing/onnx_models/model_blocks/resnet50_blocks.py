import os.path
from typing import List, Tuple, Optional
from collections import namedtuple, defaultdict
import tempfile
import onnx
import hidet
from hidet.utils import hidet_cache_file
from ..utils import export_torch_to_onnx

try:
    import torch
    from torch import nn
    import torchvision.models
except ImportError:
    pass


class ConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        return x


def conv_bn_relu(batch_size, height, width, in_channels, out_channels, kernel_size, stride, padding, bias=True) -> onnx.ModelProto:
    module = ConvBnRelu(in_channels, out_channels, kernel_size, stride, padding, bias)
    module.eval()
    x = torch.randn([batch_size, in_channels, height, width], dtype=torch.float32)
    module(x)

    _, path = tempfile.mkstemp()

    torch.onnx.export(module,
                      args=x,
                      f=path,
                      training=torch.onnx.TrainingMode.PRESERVE,
                      input_names=['x'],
                      output_names=['y'],
                      opset_version=12,
                      dynamic_axes={
                          'x': {0: 'bs'},
                          'y': {0: 'bs'}
                      },
                      do_constant_folding=False)
    onnx.checker.check_model(path)
    onnx_model = onnx.load_model(path)
    return onnx_model


Conv2dConfig = namedtuple('Conv2dConfig', field_names=['batch_size', 'height', 'width', 'in_channels', 'out_channels', 'kernel_size', 'stride', 'padding'])


def get_resnet50_configs(batch_size: int = 1) -> List[Conv2dConfig]:
    resnet50 = torchvision.models.resnet50()
    config_count = defaultdict(int)

    def hook(module: nn.Module, inputs: Tuple[torch.Tensor]):
        if isinstance(module, nn.Conv2d):
            c = module
            x = inputs[0]
            w = module.weight
            assert isinstance(x, torch.Tensor)
            config = Conv2dConfig(
                batch_size=x.size(0), height=x.size(2), width=x.size(3), in_channels=x.size(1),
                out_channels=w.size(0), kernel_size=(w.size(2), w.size(3)), stride=c.stride, padding=c.padding
            )
            config_count[config] += 1
            # print(config)

    def register_hook(module: nn.Module):
        module.register_forward_pre_hook(hook)

    resnet50.apply(register_hook)
    resnet50(torch.randn(batch_size, 3, 224, 224))
    # for a, b in config_count.items():
    # print(b, a)
    # print(a, b)
    # as of Python 3.6, the order of dict keys is the insertion order in CPython.
    return list(config_count.keys())


def print_implicit_gemm_workloads(configs: List[Conv2dConfig] = None):
    if configs is None:
        configs = get_resnet50_configs()
    for idx, config in enumerate(configs):
        n, c, h, w = config.batch_size, config.in_channels, config.height, config.width
        oc = config.out_channels
        kx, ky = config.kernel_size
        px, py = config.padding
        sx, sy = config.stride
        oh, ow = (h + px * 2 - kx) // sx + 1, (w + py * 2 - ky) // sy + 1
        m_size = n * oh * ow
        n_size = oc
        k_size = kx * ky * c
        print(m_size, n_size, k_size)


def conv_bn_relu_onnx_path(idx: int) -> str:
    path = hidet.utils.hidet_cache_file('onnx', f'conv_{idx}.onnx')
    if not os.path.exists(path):
        export_conv_bn_relu()
    if not os.path.exists(path):
        raise ValueError('failed generate onnx model')
    return path


def conv_bn_relu_input_shape(bs: int, idx: int) -> List[int]:
    shapes = {
        0: [3, 224, 224],
        1: [64, 56, 56],
        2: [64, 56, 56],
        3: [64, 56, 56],
        4: [256, 56, 56],
        5: [256, 56, 56],
        6: [128, 56, 56],
        7: [128, 28, 28],
        8: [256, 56, 56],
        9: [512, 28, 28],
        10: [128, 28, 28],
        11: [512, 28, 28],
        12: [256, 28, 28],
        13: [256, 14, 14],
        14: [512, 28, 28],
        15: [1024, 14, 14],
        16: [256, 14, 14],
        17: [1024, 14, 14],
        18: [512, 14, 14],
        19: [512, 7, 7],
        20: [1024, 14, 14],
        21: [2048, 7, 7],
        22: [512, 7, 7],
    }
    return [bs] + shapes[idx]


def get_resnet50_block(name: str, batch_size=1, precision='float32', nocache=False) -> Tuple[str, List[str], List["hidet.Tensor"]]:
    assert precision == 'float32'
    a, b, c = name.split('_')  # resnet50_conv_0 to resnet50_conv_22
    conv_idx = int(c)
    configs = get_resnet50_configs(batch_size)
    config = configs[conv_idx]
    x_shape = conv_bn_relu_input_shape(batch_size, conv_idx)
    model = ConvBnRelu(in_channels=config.in_channels, out_channels=config.out_channels, kernel_size=config.kernel_size, stride=config.stride, padding=config.padding, bias=True)

    x = torch.randn(x_shape)
    return export_torch_to_onnx(
        onnx_path=hidet_cache_file('onnx', 'resnet50', f'{name}_bs{batch_size}.onnx'),
        model=model,
        input_names=['x'],
        inputs=[x],
        nocache=nocache
    )


if __name__ == '__main__':
    for name in [
        'resnet50_conv_0',
        'resnet50_conv_1',
        'resnet50_conv_2',
        'resnet50_conv_3',
        'resnet50_conv_4',
        'resnet50_conv_5',
        'resnet50_conv_6',
        'resnet50_conv_7',
        'resnet50_conv_8',
        'resnet50_conv_9',
        'resnet50_conv_10',
        'resnet50_conv_11',
        'resnet50_conv_12',
        'resnet50_conv_13',
        'resnet50_conv_14',
        'resnet50_conv_15',
        'resnet50_conv_16',
        'resnet50_conv_17',
        'resnet50_conv_18',
        'resnet50_conv_19',
        'resnet50_conv_20',
        'resnet50_conv_21',
        'resnet50_conv_22',
    ]:
        get_resnet50_block(name)
