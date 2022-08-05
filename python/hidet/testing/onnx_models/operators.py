from typing import List, Tuple
from .utils import export_torch_to_onnx
import hidet
from hidet.utils import hidet_cache_file

try:
    import torch
    from torch import nn
except ImportError:
    pass


class ReduceSum(nn.Module):
    def __init__(self, dims: List[int], keepdim=True):
        super().__init__()
        self.dims = dims
        self.keepdim = keepdim

    def forward(self, x: torch.Tensor):
        return x.sum(self.dims, self.keepdim)


class Matmul(nn.Module):
    def __init__(self, layout: str):
        super().__init__()
        assert layout in ['NN', 'NT', 'TN', 'TT']
        self.layout = layout

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        if self.layout[0] == 'T':
            x = torch.transpose(x, -1, -2)
        if self.layout[1] == 'T':
            y = torch.transpose(y, -1, -2)
        return torch.matmul(x, y)


def get_onnx_operator(name: str, batch_size=1, precision='float32') -> Tuple[str, List[str], List["hidet.Tensor"]]:
    assert precision == 'float32'
    onnx_path = hidet_cache_file('onnx', 'op', f'{name}.onnx')
    if name.startswith('op_sum_'):
        a, b, c = name.split('_')   # op_sum_0
        op_idx = int(c)
        idx_2_configs = {
            0: [[batch_size, 8, 128, 768], [1], False],
            1: [[batch_size, 8, 128, 768], [3], False],
        }
        shape, dims, keepdim = idx_2_configs[op_idx]
        return export_torch_to_onnx(
            onnx_path=onnx_path,
            model=ReduceSum(dims=dims, keepdim=keepdim),
            input_names=['x'],
            inputs=[torch.randn(shape)],
        )
    elif name.startswith('op_resnet50_conv'):
        a, b, c, d = name.split('_')
        op_idx = int(d)
        idx_2_configs = {
            2: [[batch_size, 256, 28, 28], 256, 3, 1, 2],
        }
        x_shape, out_channels, kernel, padding, strides = idx_2_configs[op_idx]
        return export_torch_to_onnx(
            onnx_path=onnx_path,
            model=nn.Conv2d(in_channels=x_shape[1], out_channels=out_channels, kernel_size=kernel, stride=strides, padding=padding, bias=False),
            input_names=['x'],
            inputs=[torch.randn(x_shape)]
        )
    elif name.startswith('op_matmul_'):     # like 'op_matmul_nn_0'
        a, b, layout, idx = name.split('_')
        layout = str(layout).upper()
        workloads = {
            0: [batch_size, 128, 128, 64],
            1: [batch_size, 128, 768, 2304],
            2: [batch_size, 128, 768, 2304],
            3: [batch_size, 128, 768, 2304],
            4: [batch_size, 2048, 2048, 2048],
            5: [batch_size, 2039, 2039, 2039],
            6: [batch_size, 2047, 2047, 2047],
            7: [batch_size, 2046, 2046, 2046],
            8: [batch_size, 2045, 2045, 2045],
            9: [batch_size, 2044, 2044, 2044],
            10: [batch_size, 2043, 2043, 2043],
            11: [batch_size, 2042, 2042, 2042],
        }
        batch_size, m_size, n_size, k_size = workloads[int(idx)]
        x = torch.randn([batch_size, m_size, k_size])
        y = torch.randn([batch_size, k_size, n_size])
        if layout[0] == 'T':
            x = torch.transpose(x, -1, -2)
        if layout[1] == 'T':
            y = torch.transpose(y, -1, -2)
        return export_torch_to_onnx(
            onnx_path=onnx_path,
            model=Matmul(layout),
            input_names=['x', 'y'],
            inputs=[x, y],
        )
    else:
        raise ValueError('')
