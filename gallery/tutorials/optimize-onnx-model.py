"""
.. currentmodule:: hidet
.. _Run ONNX Model with Hidet:

Optimize ONNX Model
===================

This tutorial walks through the steps to run a model in `ONNX format <https://onnx.ai/>`_ with Hidet.
The ResNet50 onnx model exported from PyTorch model zoo would be used as an example.
"""

# %%
# Preparation of ONNX model
# -------------------------
# We first export the pretrained resnet50 model from torchvision model zoo to an onnx model, using
# :external:func:`torch.onnx.export`. After exporting, there will be a file named ``resnet50.onnx``
# under current working directory.

import os
import torch

# the path to save the onnx model
onnx_path = './resnet50.onnx'

# load pretrained resnet50 and create a random input
torch_model = torch.hub.load(
    'pytorch/vision:v0.9.0', 'resnet50', pretrained=True, verbose=False
)
torch_model = torch_model.cuda().eval()
torch_data = torch.randn([1, 3, 224, 224]).cuda()

# export the pytorch model to onnx model 'resnet50.onnx'
torch.onnx.export(
    model=torch_model,
    args=torch_data,
    f=onnx_path,
    input_names=['data'],
    output_names=['output'],
    dynamic_axes={'data': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
)

print('{}: {:.1f} MiB'.format(onnx_path, os.path.getsize(onnx_path) / (2**20)))

# %%
# Before going further, we first measure the latency of reset50 directly using PyTorch for inference.
# The :func:`benchmark_func() <hidet.utils.benchmark_func>` function runs the given function multiple times to
# get the median latency.

from hidet.utils import benchmark_func

print('PyTorch: {:.3f} ms'.format(benchmark_func(lambda: torch_model(torch_data))))

# %%
# Load the onnx model with Hidet
# ------------------------------
# To run the onnx model, we should first load the model with :func:`hidet.graph.frontend.from_onnx` function by giving
# the path to the onnx model. This function returns callable object, which applies all operators in the onnx model to
# the input argument and returns the output tensor(s). The onnx model can be dynamic-shaped (e.g., in this example, the
# batch size is dynamic).
import numpy as np
import hidet

# load onnx model 'resnet50.onnx'
hidet_onnx_module = hidet.graph.frontend.from_onnx(onnx_path)

print('Input names:', hidet_onnx_module.input_names)
print('Output names: ', hidet_onnx_module.output_names)

# %%
# Imperatively run the model
# --------------------------
# To run the model, we first create a hidet tensor from torch tensor with :func:`hidet.from_torch`. We directly
# call ``hidet_onnx_module`` to apply the operators in loaded onnx model to the given input tensor and get the output
# tensor.

# create a hidet tensor from pytorch tensor.
data: hidet.Tensor = hidet.from_torch(torch_data)

# apply the operators in onnx model to given 'data' input tensor
output: hidet.Tensor = hidet_onnx_module(data)

# check the output of hidet with pytorch
torch_output = torch_model(torch_data).detach()
np.testing.assert_allclose(
    actual=output.cpu().numpy(), desired=torch_output.cpu().numpy(), rtol=1e-2, atol=1e-2
)

# %%
# Trace the model and run
# -----------------------
# A more efficient way to run the model is to first trace the execution and get the static computation graph of the deep
# learning model. We can use :func:`hidet.symbol_like` to create a symbol tensor. We can get the symbol tensor output by
# running the model with the symbol tensor as input. The output is a symbol tensor that contains all information of how
# it is derived. We can use :func:`hidet.trace_from` to create the static computation graph from the symbol output
# tensor. In hidet, we use :class:`hidet.graph.FlowGraph` to represent such a computation graph, and it is also the
# basic unit of graph-level optimizations.

symbol_data = hidet.symbol_like(data)
symbol_output = hidet_onnx_module(symbol_data)
graph: hidet.FlowGraph = hidet.trace_from(symbol_output)


# %%
# We can directly call the flow graph to run it. A more efficient way is to create a
# CUDA Graph according to the flow graph and run the CUDA Graph.
#
# .. note::
#   :class: margin
#
#   The `CUDA Graph <https://developer.nvidia.com/blog/cuda-graphs/>`_ is a more efficient
#   way to submit workload to NVIDIA GPU, it eliminates most of the framework-side overhead.
#
# We use :meth:`~hidet.graph.FlowGraph.cuda_graph` method of a :class:`~hidet.graph.FlowGraph` to create a
# :class:`~hidet.cuda.graph.CudaGraph`.
# Then, we use :meth:`~hidet.cuda.graph.CudaGraph.run` method to run the cuda graph.


def bench_hidet_graph(graph: hidet.FlowGraph):
    cuda_graph = graph.cuda_graph()
    (output,) = cuda_graph.run([data])
    np.testing.assert_allclose(
        actual=output.cpu().numpy(),
        desired=torch_output.cpu().numpy(),
        rtol=1e-2,
        atol=1e-2,
    )
    print('  Hidet: {:.3f} ms'.format(benchmark_func(lambda: cuda_graph.run())))


bench_hidet_graph(graph)

# %%
# Optimize FlowGraph
# ------------------
# To optimize the model, we set the level of operator schedule space to 2 with :func:`hidet.option.search_space`. We also
# conduct graph level optimizations with :func:`hidet.graph.optimize`.

# Set the search space level for kernel tuning. By default, the search space level is 0, which means no kernel tuning.
# There are three choices: 0, 1, and 2. The higher the level, the better performance but the longer compilation time.
hidet.option.search_space(0)

# optimize the flow graph, such as operator fusion
with hidet.graph.PassContext() as ctx:
    ctx.save_graph_instrument('./outs/graphs')
    graph_opt: hidet.FlowGraph = hidet.graph.optimize(graph)

bench_hidet_graph(graph_opt)

# %%
# Summary
# -------
# Hidet is a DNN inference framework that accepts ONNX model. It conducts both graph-level and operator-level
# optimizations. We follow the following steps to run an ONNX model in Hidet:
#
# 1. Load the model with :func:`hidet.graph.frontend.from_onnx`.
# 2. Run the model with symbolic inputs, and use :func:`hidet.trace_from` to create the :class:`hidet.graph.FlowGraph`.
# 3. Create a :class:`hidet.cuda.graph.CudaGraph` using :func:`hidet.graph.FlowGraph.cuda_graph`.
# 4. Run the cuda graph.
