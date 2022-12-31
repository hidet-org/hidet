# Hidet: A compilation-based DNN inference framework
[Documentation](http://docs.hidet.org:9000/)

Hidet is an open-source DNN inference framework based on compilation. It takes an ONNX model as input, conducts a series 
of graph-level and operator-level optimizations, and does inference. 

## Getting Started

### Installation
```bash
pip install hidet
```
See [here](http://docs.hidet.org:9000/) for building from source.

### Usage

Optimize a PyTorch model through hidet (require PyTorch 2.0):
```python
import torch
import hidet

hidet.torch.register_dynamo_backends()  # register hidet backends for pytorch dynamo

model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
model_opt = torch.compile(model, backend='hidet')  # Compile the model through Hidet
y = model_opt(torch.randn(1, 3, 224, 224, device='cuda'))
```
See the following tutorials for more details:
- [Optimize PyTorch models](http://docs.hidet.org:9000/gallery/tutorials/optimize-pytorch-model.html)
- [Optimize ONNX models](http://docs.hidet.org:9000/gallery/tutorials/run-onnx-model.html)

## License
Hidet is released under the [Apache 2.0 license](LICENSE).

## Publication
Hidet originates from the following [paper](https://arxiv.org/abs/2210.09603). If you use Hidet in your research, 
welcome to cite the paper.
```text
@misc{hidet,
  title = {Hidet: Task Mapping Programming Paradigm for Deep Learning Tensor Programs},
  author = {Ding, Yaoyao and Yu, Cody Hao and Zheng, Bojian and Liu, Yizhi and Wang, Yida and Pekhimenko, Gennady},
  doi = {10.48550/ARXIV.2210.09603},
  url = {https://arxiv.org/abs/2210.09603},
  publisher = {arXiv},
  year = {2022},
}
```
