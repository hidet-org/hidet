import numpy as np
import hidet
from hidet.testing.utils import benchmark_func

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torchinfo import summary

h = 280
w = 280

c0 = 3
c1 = 16
c2 = 8

flattened = 39200
latent = 2000

batch_size = 1

device = 'cuda'
torch.manual_seed(0)
onnx_path = './torch_model.onnx'


def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class Autoencoder(nn.Module):
    def __init__(self, h, w, c0, c1, c2, latent):
        super(Autoencoder, self).__init__()

        self.conv1 = nn.Conv2d(c0, c1, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c1)
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=3, stride=4, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(flattened, latent, bias=False)
        self.fc2 = nn.Linear(latent, flattened, bias=False)
        self.convtranspose1 = nn.ConvTranspose2d(c2, c2, kernel_size=3, stride=1, padding=1, output_padding=0,
                                                 bias=False)
        self.bn3 = nn.BatchNorm2d(c2)
        self.convtranspose2 = nn.ConvTranspose2d(c2, c1, kernel_size=3, stride=4, padding=0, output_padding=1,
                                                 bias=False)
        self.bn4 = nn.BatchNorm2d(c1)
        self.convtranspose3 = nn.ConvTranspose2d(c1, c0, kernel_size=3, stride=1, padding=1, output_padding=0,
                                                 bias=False)
        self.apply(_weights_init)

    def forward(self, x):
        x = self.convtranspose1(x)
        return x
        x = self.bn1(F.leaky_relu(self.conv1(x)))
        x = self.bn2(F.leaky_relu(self.conv2(x)))
        shape = x.shape
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.view(shape)
        x = self.convtranspose1(x)
        # x = self.bn3(F.leaky_relu(self.convtranspose1(x)))
        # x = self.bn4(F.leaky_relu(self.convtranspose2(x)))
        # x = self.convtranspose3(x)

        return x


if __name__ == '__main__':
    torch_model = Autoencoder(h, w, c0, c1, c2, latent).to(device)
    print(torch_model)

    # summary(torch_model, (batch_size, c0, h, w), device=device)

    # torch_data = torch.randn([batch_size, c0, h, w]).cuda()
    torch_data = torch.randn([1, 8, 70, 70]).cuda()
    print('* Saving to ' + onnx_path + '...')
    torch_model = torch_model.eval()
    torch.onnx.export(torch_model, torch_data, onnx_path, verbose=False)
    print('done!')

    print('* Importing onnx graph...')
    hidet_onnx_module = hidet.graph.frontend.from_onnx(onnx_path)
    print('Input names:', hidet_onnx_module.input_names)
    print('Output names: ', hidet_onnx_module.output_names)
    print('done!')

    print('* Creating hidet tensor...')
    data: hidet.Tensor = hidet.from_torch(torch_data)
    print('done!')

    print('* Start PyTorch inference...')
    torch_output = torch_model(torch_data).detach()
    print('done!')

    print('* Start Hidet (from onnx) inference...')
    hidet_graph: hidet.FlowGraph = hidet_onnx_module.flow_graph_for([data])
    hidet.option.search_space(0)
    with hidet.graph.PassContext() as ctx:
        ctx.save_graph_instrument(out_dir='./outs/graph')
        hidet_graph_opt = hidet.graph.optimize(hidet_graph)
    cuda_graph = hidet_graph_opt.cuda_graph()
    output = cuda_graph.run_with_inputs([data])[0]
    with open('hidet_graph_opt.json', 'w') as f:
        hidet.utils.netron.dump(hidet_graph_opt, f)
    print('done!')

    print('* Benchmarking...')
    print('  Hidet (from onnx): {:.3f} ms'.format(benchmark_func(lambda: cuda_graph.run())))
    print('  PyTorch: {:.3f} ms'.format(benchmark_func(lambda: torch_model(torch_data))))
    np.testing.assert_allclose(actual=output.numpy(), desired=torch_output.cpu().numpy(), rtol=1e-2, atol=1e-2)
