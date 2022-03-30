import torch
from torchvision.models.resnet import resnet50
from hidet.utils import Timer
from hidet.ffi.cuda_api import CudaAPI


def demo_resnet50():
    model = resnet50().cuda()
    model.train(False)
    x = torch.rand(32, 3, 224, 224).cuda()
    for t in range(3):
        with Timer(f'torch {t}'):
            y = model(x)
            CudaAPI.device_synchronization()
        y = None


if __name__ == '__main__':
    demo_resnet50()
