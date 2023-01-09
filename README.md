# Hidet: A compilation-based deep learning framework
[**Documentation**](http://docs.hidet.org/) 

![GitHub](https://img.shields.io/github/license/hidet-org/hidet)
![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/hidet-org/hidet/tests.yaml)


Hidet is an open-source DNN inference framework based on compilation. 
It supports end-to-end compilation of DNN models from PyTorch and ONNX to efficient cuda kernels.
A series of graph-level and operator-level optimizations are applied to optimize the performance.

## Getting Started

### Installation
```bash
pip install hidet
```
See [here](http://docs.hidet.org/) for building from source.

### Usage

Optimize a PyTorch model through hidet (require PyTorch 2.0):
```python
import torch
import hidet

# Register hidet backends for pytorch dynamo, can be omitted if you import torch before hidet
hidet.torch.register_dynamo_backends()  

# Define pytorch model
model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True).cuda().eval()
x = torch.rand(1, 3, 224, 224).cuda()

# Compile the model through Hidet
model_opt = torch.compile(model, backend='hidet')  

# Run the optimized model
y = model_opt(x)
```
See the following tutorials to learn other usgae:
- [Quick Start](http://docs.hidet.org/stable/gallery/getting-started/quick-start.html)
- [Optimize PyTorch models](http://docs.hidet.org/stable/gallery/tutorials/optimize-pytorch-model.html)
- [Optimize ONNX models](http://docs.hidet.org/stable/gallery/tutorials/run-onnx-model.html)

## Publication
Hidet originates from the following research work. If you used **Hidet** in your research, welcome to cite our
[paper](https://arxiv.org/abs/2210.09603). 

- **Hidet: Task-Mapping Programming Paradigm for Deep Learning Tensor Programs.**  
  Yaoyao Ding, Cody Hao Yu, Bojian Zheng, Yizhi Liu, Yida Wang, and Gennady Pekhimenko. 

## Development 
Hidet is currently under active development by a team at [CentML Inc](https://centml.ai/). 

## Contributing
We welcome contributions from the community. Please see 
[contribution guide](https://docs.hidet.org/stable/developer-guides/contributing.html)
for more details.

## License
Hidet is released under the [Apache 2.0 license](LICENSE).
