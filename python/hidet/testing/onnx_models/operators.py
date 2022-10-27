from typing import List, Tuple
from hidet.utils import hidet_cache_file
from .utils import export_torch_to_onnx

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


class DepthwiseConv2d(nn.Module):
    def __init__(self, c, s):
        super().__init__()
        self.c = c
        self.s = s

    def forward(self, x: torch.Tensor, w: torch.Tensor):
        return torch.conv2d(x, w, bias=None, stride=self.s, padding=0, groups=self.c)


def get_onnx_operator(name: str, batch_size=1, precision='float32') -> Tuple[str, List[str], List["hidet.Tensor"]]:
    onnx_path = hidet_cache_file('onnx', 'op', f'bs{batch_size}_{precision}_{name}.onnx')
    if name.startswith('op_sum_'):
        _, _, c = name.split('_')  # op_sum_0
        op_idx = int(c)
        idx_2_configs = {0: [[batch_size, 8, 128, 768], [1], False], 1: [[batch_size, 8, 128, 768], [3], False]}
        shape, dims, keepdim = idx_2_configs[op_idx]
        return export_torch_to_onnx(
            onnx_path=onnx_path,
            model=ReduceSum(dims=dims, keepdim=keepdim),
            input_names=['x'],
            inputs=[torch.randn(shape)],
            precision=precision,
        )
    elif name.startswith('op_resnet50_conv'):
        _, _, _, d = name.split('_')
        op_idx = int(d)
        idx_2_configs = {2: [[batch_size, 256, 28, 28], 256, 3, 1, 2]}
        x_shape, out_channels, kernel, padding, strides = idx_2_configs[op_idx]
        return export_torch_to_onnx(
            onnx_path=onnx_path,
            model=nn.Conv2d(
                in_channels=x_shape[1],
                out_channels=out_channels,
                kernel_size=kernel,
                stride=strides,
                padding=padding,
                bias=False,
            ),
            input_names=['x'],
            inputs=[torch.randn(x_shape)],
            precision=precision,
        )
    elif name.startswith('op_matmul_'):  # like 'op_matmul_nn_0'
        _, _, layout, idx = name.split('_')
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
            onnx_path=onnx_path, model=Matmul(layout), input_names=['x', 'y'], inputs=[x, y], precision=precision
        )
    elif name.startswith('op_setgan_conv_'):  # like 'op_setgan_conv_3'
        _, _, _, idx = name.split('_')
        workload = {
            # idx: [batch_size, in_channels, height, width, out_channels, kx, ky, sx, sy, px, py]
            0: [576, 1, 64, 64, 32, 4, 4, 2, 2, 0, 0],
            1: [576, 32, 31, 31, 64, 4, 4, 2, 2, 0, 0],
            2: [576, 64, 14, 14, 128, 4, 4, 2, 2, 0, 0],
            3: [576, 128, 6, 6, 128, 3, 3, 2, 2, 1, 1],
            4: [448, 128, 6, 6, 128, 4, 4, 2, 2, 0, 0],
            5: [576, 128, 6, 6, 128, 4, 4, 2, 2, 0, 0],
            6: [576, 128, 6, 6, 128, 1, 1, 1, 1, 0, 0],
            7: [448, 128, 6, 6, 128, 1, 1, 1, 1, 0, 0],
            8: [448, 64, 14, 14, 64, 4, 4, 2, 2, 0, 0],
            9: [576, 64, 14, 14, 64, 4, 4, 2, 2, 0, 0],
            10: [576, 64, 14, 14, 64, 1, 1, 1, 1, 0, 0],
            11: [448, 64, 14, 14, 64, 1, 1, 1, 1, 0, 0],
            12: [448, 32, 31, 31, 32, 4, 4, 2, 2, 0, 0],
            13: [576, 32, 31, 31, 32, 4, 4, 2, 2, 0, 0],
            14: [576, 32, 31, 31, 32, 1, 1, 1, 1, 0, 0],
            15: [448, 32, 31, 31, 32, 1, 1, 1, 1, 0, 0],
        }
        bs, in_channels, height, width, out_channels, kx, ky, sx, sy, px, py = workload[int(idx)]
        return export_torch_to_onnx(
            onnx_path=onnx_path,
            model=nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(kx, ky),
                stride=(sx, sy),
                padding=(px, py),
                bias=True,
            ),
            input_names=['x'],
            inputs=[torch.randn(bs, in_channels, height, width)],
            precision=precision,
        )
    elif name.startswith('op_gemm_'):  # like 'op_gemm_m_n_k'
        _, _, m, n, k = name.split('_')
        m, n, k = int(m), int(n), int(k)
        return export_torch_to_onnx(
            onnx_path=onnx_path,
            model=Matmul(layout='NN'),
            input_names=['x', 'y'],
            inputs=[torch.randn(batch_size, m, k), torch.randn(batch_size, k, n)],
            precision=precision,
        )
    elif name.startswith('op_dwc_'):  # like 'op_dwc_n_c_h_w_s_k'
        _, _, n, c, h, w, s, k = name.split('_')
        n, c, h, w, s, k = int(n), int(c), int(h), int(w), int(s), int(k)
        return export_torch_to_onnx(
            onnx_path=onnx_path,
            model=DepthwiseConv2d(c, s),
            input_names=['x', 'w'],
            inputs=[torch.randn(n, c, (h - 1) * s + k, (w - 1) * s + k), torch.randn(c, 1, k, k)],
            precision=precision,
        )
    else:
        raise ValueError('')
