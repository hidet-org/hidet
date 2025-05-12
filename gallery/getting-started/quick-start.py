"""
.. currentmodule:: hidet

Quick Start
===========

This guide walks through the key functionality of Hidet for tensor computation.
"""

# %%
# Optimize PyTorch model with Hidet
# ---------------------------------
# .. note::
#   :class: margin
#
#   ``torch.compile(...)`` requires PyTorch 2.3+.
#
# The easiest way to use Hidet is to use the :func:`torch.compile` function with ``hidet`` as the backend, such as
#
# .. code-block:: python
#
#   model_opt = torch.compile(model, backend='hidet')
#
# Next, we use resnet18 model as an example to show how to optimize a PyTorch model with Hidet.
#
# .. tip::
#   :class: margin
#
#   Because tf32 is enabled by default for torch's cudnn backend, the torch's precision is slightly low.
#   You could disable the tf32 (See also `PyTorch TF32`_).
# .. _PyTorch TF32: https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices

import hidet
import torch

# take resnet18 as an example
x = torch.randn(1, 3, 224, 224, dtype=torch.float16).cuda()
model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True, verbose=False)
model = model.cuda().eval().to(torch.float16)

# optimize the model with 'hidet' backend
model_opt = torch.compile(model, backend='hidet', mode='max-autotune')

# run the optimized model
y1 = model_opt(x)
y2 = model(x)

# check the correctness
torch.testing.assert_close(actual=y1, expected=y2, rtol=2e-2, atol=2e-2)


# benchmark the performance
for name, model in [('eager', model), ('hidet', model_opt)]:
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start_event.record()
    for _ in range(100):
        y = model(x)
    end_event.record()
    torch.cuda.synchronize()
    print('{:>10}: {:.3f} ms'.format(name, start_event.elapsed_time(end_event) / 100.0))

# %%
# One operator can have multiple equivalent implementations (i.e., kernel programs) with different performance. We
# usually need to try different implementations for each concrete input shape to find the best one for the specific
# input shape. This process is called `kernel tuning`. To enable kernel tuning, we can use the following config in
# hidet:
#
# .. code-block:: python
#
#    # 0 - no tuning, default kernel will be used
#    # 1 - tuning in a small search space
#    # 2 - tuning in a large search space, will take longer time and achieves better performance
#    hidet.torch.dynamo_config.search_space(2)
#
# When kernel tuning is enabled, hidet can achieve the following performance on NVIDIA RTX 4090:
#
# .. code-block:: text
#
#    eager: 1.176 ms
#    hidet: 0.286 ms
#


# %%
# Hidet provides some configurations to control the optimization of hidet backend. such as
#
# - **Search Space**: you can choose the search space of operator kernel tuning. A larger schedule space usually
#   achieves the better performance, but takes longer time to optimize.
# - **Correctness Checking**: print the correctness checking report. You can know the numerical difference between the
#   hidet generated operator and the original pytorch operator.
# - **Other Configurations**: you can also configure the other optimizations of hidet backend, such as using a lower
#   precision of data type automatically (e.g., float16), or control the behavior of parallelization of the reduction
#   dimension of the matrix multiplication and convolution operators.
#
# In the remaining parts, we will show you the key components of Hidet.
#
#
# Define tensors
# --------------
#
# .. tip::
#   :class: margin
#
#   Besides :func:`~hidet.randn`, we can also use :func:`~hidet.zeros`, :func:`~hidet.ones`, :func:`~hidet.full`,
#   :func:`~hidet.empty` to create tensors with different initialized values. We can use :func:`~hidet.from_torch` to
#   convert a PyTorch tensor to Hidet tensor that shares the same memory. We can also use :func:`~hidet.asarray` to
#   convert python list or numpy ndarray to Hidet tensor.
#
# A *tensor* is a n-dimension array. As other machine learning framework,
# Hidet takes :class:`~hidet.Tensor` as the core object to compute and manipulate.
# The following code defines a tensor with randomly initialized tensor with :func:`hidet.randn`.

a = hidet.randn([2, 3], device='cuda')
print(a)

# %%
# Each :class:`~hidet.Tensor` has :attr:`~hidet.Tensor.dtype` to define the type of each tensor element,
# and :attr:`~hidet.Tensor.device` to tell which device this tensor resides on, and
# :attr:`~hidet.Tensor.shape` to indicate the size of each dimension. The example defines a ``float32`` tensor on
# ``cuda`` device with shape ``[2, 3]``.
#

# %%
# Run operators
# -------------
# Hidet provides :mod:`a bunch of operators <hidet.ops>` (e.g., :func:`~hidet.ops.matmul` and
# :func:`~hidet.ops.conv2d`) to compute and manipulate tensors. We can do a matrix multiplication as follows:
b = hidet.randn([3, 2], device='cuda')
c = hidet.randn([2], device='cuda')
d = hidet.ops.matmul(a, b)
d = d + c  # 'd + c' is equivalent to 'hidet.ops.add(d, c)'
print(d)

# %%
# In this example, the operator is executed on the device at the time we call it, thus it is in an `imperative` style
# of execution. Imperative execution is intuitive and easy to debug. But it prevents some graph-level optimization
# opportunities and suffers from higher kernel dispatch latency.
#
# In the next section, we would introduce another way to execute operators.
#
# Symbolic tensor and flow graph
# ------------------------------
# In hidet, each tensor has an optional :attr:`~hidet.Tensor.storage` attribute that represents a block of
# memory that stores the contents of the tensor. If the storage attribute is None, the tensor is a `symbolic` tensor.
# We could use :func:`hidet.symbol_like` or :func:`hidet.symbol` to create a symbolic tensor. Symbolic tensors are
# returned if any input tensor of an operator is symbolic. We could know how the symbolic tensor is computed via the
# :attr:`~hidet.Tensor.trace` attribute. It is a tuple ``(op, idx)`` where ``op`` is the operator produces this
# tensor and ``idx`` is the index of this tensor in the operator's outputs.


def linear_bias(x, b, c):
    return hidet.ops.matmul(x, b) + c


x = hidet.symbol_like(a)
y = linear_bias(x, b, c)

assert x.trace is None
assert y.trace is not None

print('x:', x)
print('y:', y)

# %%
# We can use trace attribute to construct the computation graph, starting from the symbolic output tensor(s).
# This is what function :func:`hidet.trace_from` does. In hidet, we use :class:`hidet.graph.FlowGraph` to
# represent the data flow graph (a.k.a, computation graph).
graph: hidet.FlowGraph = hidet.trace_from(y)
print(graph)


# %%
# Optimize flow graph
# -------------------
# .. tip::
#   :class: margin
#
#   We may config optimizations with :class:`~hidet.graph.PassContext`.
#   Potential configs:
#
#   - Whether to use tensor core.
#   - Whether to use low-precision data type (e.g., ``float16``).
#
# Flow graph is the basic unit of graph-level optimizations in hidet. We can optimize a flow graph with
# :func:`hidet.graph.optimize`. This function applies the predefined passes to optimize given flow graph.
# In this example, we fused the matrix multiplication and element-wise addition into a single operator.

opt_graph: hidet.FlowGraph = hidet.graph.optimize(graph)
print(opt_graph)

# %%
# Run flow graph
# --------------
# We can directly call the flow graph to run it:
y1 = opt_graph(a)
print(y1)

# %%
# For CUDA device, a more efficient way is to create a cuda graph to dispatch the kernels in a flow graph
# to the NVIDIA GPU.
cuda_graph = opt_graph.cuda_graph()
outputs = cuda_graph.run([a])
y2 = outputs[0]
print(y2)

# %%
# Summary
# -------
# In this quick start guide, we walk through several important functionalities of hidet:
#
# - Define tensors.
# - Run operators imperatively.
# - Use symbolic tensor to create computation graph (e.g., flow graph).
# - Optimize and run flow graph.
#
