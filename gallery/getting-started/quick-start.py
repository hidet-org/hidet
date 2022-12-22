"""
.. currentmodule:: hidet

Quick Start
===========

This guide walks through the key functionality of Hidet for tensor computation.
"""
import torch._dynamo as dynamo
import torch.hub

# %%
# We should first import hidet.
import hidet

# %%
# Optimize PyTorch model with Hidet
# ---------------------------------
# .. tip::
#   :class: margin
#
#   Torch dynamo is a feature introduced in PyTorch 2.0, which has not been officially released yet. Please install the
#   nightly build of PyTorch to use this feature.
#
# The easiest way to use Hidet is to use the :func:`torch.compile` function with 'hidet' as the backend.

# use a pretrained model resnet18 as an example
x = torch.randn(1, 3, 224, 224).cuda()
model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True).cuda().eval()


# first optimize the model with pytorch's default backend 'inductor'
with torch.no_grad():
    model_inductor = torch.compile(model, backend='inductor')
    model_inductor(x)

    # benchmark the performance of the optimized model with inductor backend
    # the benchmark_func would run (warmup + number * repeat) times and return the median latency.
    from hidet.testing import benchmark_func
    inductor_latency: float = benchmark_func(lambda: model_inductor(x), warmup=3, number=10, repeat=10)

# torch dynamo requires to call the reset function before we use another backend
dynamo.reset()

# optimize the model with hidet provided backend 'hidet'
# the first run of hidet optimized model would take 20 to 30 minutes to optimize the model and tune the execution
# schedule for each convolution/matrix multiplication in the model. Usually, the optimizing time depends on the number
# of these operators in the model and the performance of your CPU. It takes about 1 minute to tune such a kernel on
# an i9-12900k CPU.
with torch.no_grad():
    model_hidet = torch.compile(model, backend='hidet')
    model_hidet(x)

    # benchmark the performance of the optimized model
    hidet_latency: float = benchmark_func(lambda: model_hidet(x), warmup=3, number=10, repeat=10)

print(' inductor: {} ms'.format(inductor_latency))
print('    hidet: {} ms'.format(hidet_latency))

# %%
# Hidet provides some configurations to control the optimization of hidet backend. You can learn more about them in
# the tutorial :doc:`/gallery/tutorials/optimize-pytorch-model`.

# %%
# Define tensors
# --------------
#
# .. tip::
#   :class: margin
#
#   Besides :func:`~hidet.randn`, we can also use :func:`~hidet.zeros`, :func:`~hidet.ones`, :func:`~hidet.full`, :func:`~hidet.empty`
#   to create tensors with different initialized values. We can use :func:`~hidet.from_torch` to convert a PyTorch tensor to Hidet tensor
#   that shares the same memory. We can also use :func:`~hidet.array` to convert python list or numpy ndarray to Hidet tensor.
#
# A *tensor* is a n-dimension array. As other machine learning framework,
# Hidet takes :class:`~hidet.graph.Tensor` as the core object to compute and manipulate.
# The following code defines a tensor with randomly initialized tensor with :func:`hidet.randn`.

a = hidet.randn([2, 3])
print(a)

# %%
# Each :class:`~hidet.graph.Tensor` has :attr:`~hidet.graph.Tensor.dtype` to define the type of each tensor element,
# and :attr:`~hidet.graph.Tensor.device` to tell which device this tensor resides on, and :attr:`~hidet.graph.Tensor.shape` to indicate
# the size of each dimension. The example defines a ``float32`` tensor on ``cuda`` device with shape ``[2, 3]``.


# %%
# Run operators
# -------------
# Hidet provides :mod:`a bunch of operators <hidet.graph.ops>` (e.g., :func:`~hidet.graph.ops.matmul` and :func:`~hidet.graph.ops.conv2d`) to compute and manipulate
# tensors. We can do a matrix multiplication as follows:
b, c = hidet.randn([3, 2]), hidet.randn([2])
d = hidet.ops.matmul(a, b)
d = d + c   # 'd + c' is equivalent to 'hidet.ops.add(d, c)'
print(d)

# %%
# In this example, the operator is executed on the device at the time we call it, thus it is in an `imperative` style of execution.
# Imperative execution is intuitive and easy to debug. But it prevents some graph-level optimization opportunities and suffers from higher
# kernel dispatch latency.
#
# In the next section, we would introduce another way to execute operators.

# %%
# Symbolic tensor and flow graph
# ------------------------------
# In hidet, each tensor has an optional :attr:`~hidet.graph.Tensor.storage` attribute that represents a block of memory that
# stores the contents of the tensor. If the storage attribute is None, the tensor is a `symbolic` tensor.
# We could use :func:`hidet.symbol_like` or :func:`hidet.symbol` to create a symbolic tensor. Symbolic tensors are returned if any
# input tensor of an operator is symbolic.
# We could know how the symbolic tensor is computed via the :attr:`~hidet.graph.Tensor.trace` attribute.
# It is a tuple ``(op, idx)`` where ``op`` is the operator produces this tensor and ``idx`` is the index of this tensor in the operator's outputs.


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
outputs = cuda_graph.run_with_inputs([a])
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

# %%
# Next Step
# ---------
# It is time to learn how to use hidet in your project. A good start is to :ref:`Run ONNX Model with Hidet`.
#

