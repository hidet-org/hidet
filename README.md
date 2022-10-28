# Hidet: A compilation-based DNN inference framework

## Introduction
Hidet is an open-source DNN inference framework based on compilation. It takes an ONNX model as input, conducts a series 
of graph-level and operator-level optimizations, and does inference. 

## Getting Started

### Installation
```bash
pip install hidet
```
Please see the documentation for installing from source code.

### Hello, world!
Adding two tensors is a good start to learn a new DNN framework.
```python
import hidet

a = hidet.randn([3, 4], device='cuda')
b = hidet.randn([3, 4], device='cuda')
print(a + b)
```
The output of this problem
```text
Compiling task add...
Tensor(shape=[3, 4], dtype='float32', device='cuda')
[[ 1.0004374   0.5608922  -0.9226169   1.4127803 ]
 [ 2.0882926  -2.9668841  -1.4881673   1.4913353 ]
 [-1.2918147   0.2576717   0.59661216 -2.0760517 ]]
```

## Documentation

See the documentation to learn how to use Hidet.

## Publication
Hidet originates from the following paper. If you use Hidet in your research, welcome to cite the paper.
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
