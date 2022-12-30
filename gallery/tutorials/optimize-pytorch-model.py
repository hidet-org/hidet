"""
Optimize PyTorch Model
======================

Hidet provides a backend to pytorch dynamo to optimize PyTorch models. To use this backend, you need to specify 'hidet'
as the backend when calling :func:`torch.compile` such as

.. code-block:: python

    # optimize the model with hidet provided backend 'hidet'
    model_hidet = torch.compile(model, backend='hidet')

.. note::
  :class: margin

  Currently, all the operators in hidet are generated by hidet itself and
  there is no dependency on kernel libraries such as cuDNN or cuBLAS. In the future, we might support to lower some
  operators to these libraries if they perform better.

Under the hood, hidet will convert the PyTorch model to hidet's graph representation and optimize the computation graph
(such as sub-graph rewrite and fusion, constant folding, etc.). After that, each operator will be lowered to hidet's
scheduling system to generate the final kernel.


Hidet provides some configurations to control the hidet backend of torch dynamo.

Search in a larger search space
-------------------------------
There are some operators that are compute-intensive and their scheduling is critical to the performance. We usually need
to search in a schedule space to find the best schedule for them to achieve the best performance on given input shapes.
However, searching in a larger schedule space usually takes longer time to optimize the model. By default, hidet will
use their default schedule to generate the kernel for all input shapes. To search in a larger schedule space to get
better performance, you can configure the search space via :func:`~hidet.graph.frontend.torch.DynamoConfig.search_space`
:

.. code-block:: python

    # There are three search spaces:
    # 0 - use default schedule, no search [Default]
    # 1 - search in a small schedule space (usually 1~30 schedules)
    # 2 - search in a large schedule space (usually more than 30 schedules)
    hidet.torch.dynamo_config.set_search_space(2)

    # After configure the search space, you can optimize the model
    model_opt = torch.compile(model, backend='hidet')

    # The actual searching happens when you first run the model to know the input shapes
    outputs = model_opt(inputs)

Please note that the search space we set through :func:`~hidet.torch.dynamo_config.set_search_space` will be read and
used when we first run the model, instead of when we call :func:`torch.compile`.

Check the correctness
---------------------
It is important to make sure the optimized model is correct. Hidet provides a configuration to print the numerical
difference between the hidet generated operator and the original pytorch operator. You can configure it via
:func:`~hidet.graph.frontend.torch.DynamoConfig.correctness_report`:

.. code-block:: python

    # enable the correctness checking
    hidet.torch.dynamo_config.correctness_report()

After enabling the correctness report, every time a new graph is received to compile, hidet will print the numerical
difference using the dummy inputs (for now, torch dynamo does not expose the actual inputs to backends, thus we can
not use the actual inputs). Let's take the resnet18 model as an example:
"""
import torch.backends.cudnn
import hidet

hidet.torch.register_dynamo_backends()  # register hidet backend to torch dynamo

x = torch.randn(1, 3, 224, 224).cuda()
model = torch.hub.load(
    'pytorch/vision:v0.9.0', 'resnet18', pretrained=True, verbose=False
)
model = model.cuda().eval()

with torch.no_grad():
    hidet.torch.dynamo_config.correctness_report()
    model_opt = torch.compile(model, backend='hidet')
    model_opt(x)

# %%
#
# .. tip::
#   :class: margin
#
#   Usually, we can expect:
#
#   - for float32: :math:`e_h \leq 10^{-5}`, and
#   - for float16: :math:`e_h \leq 10^{-2}`.
#
# The correctness report will print the harmonic mean of the absolute error and relative error for each operator:
#
# .. math::
#   e_h = \frac{|actual - expected|}{|expected| + 1} \quad (\frac{1}{e_h} = \frac{1}{e_a} + \frac{1}{e_r})
#
#
# where :math:`actual`, :math:`expected` are the actual and expected results of the operator, respectively.
# The :math:`e_a` and :math:`e_r` are the absolute error and relative error, respectively. The harmonic mean error is
# printed for each operator.
#

# %%
# Operator configurations
# -----------------------
#
# Use CUDA Graph to dispatch kernels
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Hidet provides a configuration to use CUDA Graph to dispatch kernels. CUDA Graph is a new feature in CUDA 11.0
# that allows us to record the kernel dispatches and replay them later. This feature is useful when we want to
# dispatch the same kernels multiple times. Hidet will enable CUDA Graph by default. You can disable it via
# :func:`~hidet.graph.frontend.torch.DynamoConfig.use_cuda_graph`:
#
# .. code-block:: python
#
#     # disable CUDA Graph
#     hidet.torch.dynamo_config.use_cuda_graph(False)
#
# in case you want to use PyTorch's CUDA Graph feature.
#
# Use low-precision data type
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Hidet provides a configuration to use low-precision data type. By default, hidet will use the same data type as
# the original PyTorch model. You can configure it via :func:`~hidet.graph.frontend.torch.DynamoConfig.use_fp16` and
# :func:`~hidet.graph.frontend.torch.DynamoConfig.use_fp16_reduction`:
#
# .. code-block:: python
#
#     # automatically transform the model to use float16 data type
#     hidet.torch.dynamo_config.use_fp16(True)
#
#     # use float16 data type as the accumulate data type in operators with reduction
#     hidet.torch.dynamo_config.use_fp16_reduction(True)
#
# You do not need to change the inputs feed to the model, as hidet will automatically cast the inputs to the
# configured data type automatically in the optimized model.
#
#
# Print the input graph
# ~~~~~~~~~~~~~~~~~~~~~
#
# If you are interested in the graph that PyTorch dynamo dispatches to hidet backend, you can configure hidet to
# print the graph via :func:`~hidet.graph.frontend.torch.DynamoConfig.print_input_graph`:
#
# .. code-block:: python
#
#     # print the input graph
#     hidet.torch.dynamo_config.print_input_graph(True)
#
# Because ResNet18 is a neat model without control flow, we can print the input graph to see how PyTorch dynamo
# dispatches the model to hidet backend:

# sphinx_gallery_start_ignore
import torch._dynamo as dynamo

hidet.torch.dynamo_config.correctness_report(False)  # reset
dynamo.reset()  # clear the compiled cache
# sphinx_gallery_end_ignore

with torch.no_grad():
    hidet.torch.dynamo_config.print_input_graph(True)
    model_opt = torch.compile(model, backend='hidet')
    model_opt(x)
