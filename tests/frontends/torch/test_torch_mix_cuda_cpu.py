import hidet
import hidet.testing
import torch
from torch import nn


class CopyTensorModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.w_cpu = torch.zeros(1, device='cpu')
        self.w_cuda = torch.zeros(1, device='cuda')

    def forward(self, x: torch.Tensor):
        return ((x.cpu() + self.w_cpu).cuda() + self.w_cuda).cpu().cuda()


def test_torch_mix_cuda_cpu():
    model = CopyTensorModule()
    x = torch.randn(3, 4, device='cpu')
    y = model(x)

    model_opt = torch.compile(model, backend='hidet')
    y1 = model_opt(x)

    torch.testing.assert_close(y, y1, rtol=0.0, atol=0.0)
