"""
.. currentmodule:: hidet
.. _Pipeline Inference:

Pipeline Inference
==================

This guide shows how to pipeline the data preprocessing and postprocessing with the model inference.


Motivation
----------

We usually repeating the following steps to run inference on a model:

1. Preprocess the new received input data.
2. Run the model inference.
3. Postprocess the output data and send.

We can draw the timeline of the above steps as follows:

.. code-block:: text

    GPU: |---------------------|-- inference y[i] --|----------------------|
    CPU: |-- preprocess x[i] --|----- wait GPU -----|-- postprocess y[i] --|

If the preprocessing and postprocessing are time-consuming, the GPU will be idle for a long time.

We can pipeline the preprocessing and postprocessing with the model inference to reduce the idle time of GPU.

.. code-block:: text

    GPU: |--------------------inference y[i]-------------------------|
    CPU: |-- postprocess y[i-1] --- preprocess x[i+1] --- wait GPU --|

Let :math:`t_{gpu}`, :math:`t_{prep}`, and :math:`t_{post}` be the time of the GPU inference, the CPU preprocessing,
and the CPU postprocessing, respectively.
The total time without pipeline will be

.. math::

    t_{gpu} + t_{prep} + t_{post}

while the total time with pipeline will be

.. math::

    \max(t_{gpu}, \; t_{post} + t_{prep}).


The following part will show how to pipeline the data preprocessing and postprocessing with the model inference in Hidet.


Pipeline in Hidet
-----------------

Prepare the model and data
~~~~~~~~~~~~~~~~~~~~~~~~~~

We use a 20 layers of multi-layer perceptron (MLP) as the model for the demo. The model is defined as follows:

"""
import time
from typing import List, Iterator
import numpy as np
import hidet.testing
from hidet.graph import nn, Tensor, FlowGraph
from hidet.runtime.cuda_graph import CudaGraph, cuda_graph_pipeline_execute

dimension = 1024


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        num_layers = 20
        self.linear_layers = []
        for i in range(num_layers):
            self.linear_layers.append(nn.Linear(dimension, dimension, bias=False))

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.linear_layers:
            x = layer(x)
        return x


# create a multi-layer perceptron model
model = MLP()

# %%
# We use numpy array as the raw inputs and final outputs of the model. The following code generates the input data.

# input and outputs
batch_size = 1
num_samples = 100
data: List[np.ndarray] = [np.random.randn(batch_size, dimension).astype('float32') for _ in range(num_samples)]
outputs: List[np.ndarray] = []

# %%
# To use the pipeline feature, we need to create a :class:`~hidet.runtime.cuda_graph.CudaGraph` object.
# We also measure the GPU latency to run the model.

# get the flow graph from the model
graph: FlowGraph = model.flow_graph_for(inputs=[hidet.randn([batch_size, dimension])])
# create a cuda graph from the flow graph
cuda_graph: CudaGraph = graph.cuda_graph()
# measure the latency of a single run and estimate the latency that runs all samples on GPU
single_run_latency = cuda_graph.profile(warmup=3, number=10, repeat=10)
print('Single run: {:.1f} milliseconds'.format(single_run_latency))
print('{} run: {:.1f} milliseconds'.format(num_samples, single_run_latency * num_samples))


# %%
# Input iterator and output consumer
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We use the ``input_iterator`` to mimic the fetching and preprocessing of data,
# and the ``output_consumer`` to mimic the post-processing of the output data.
# We can iterate over ``input_iterator()`` to get the input data and call ``output_consumer`` to consume the output data.
# We sleep for half of the single run latency to mimic the time of fetching and preprocessing the data.


def input_iterator() -> Iterator[List[np.ndarray]]:
    for i in range(num_samples):
        time.sleep(single_run_latency / 2000.0)
        yield [data[i]]


def output_consumer(outs: List[np.ndarray]):
    time.sleep(single_run_latency / 2000.0)
    outputs.append(outs[0])


# %%
# Run without pipeline
# ~~~~~~~~~~~~~~~~~~~~
# Without pipeline, we can run the model inference as follows:

t1 = time.time()
for inputs in input_iterator():
    # transfer the numpy input to hidet tensor on cuda
    hidet_inputs = [hidet.array(x).cuda() for x in inputs]
    # run the model inference
    hidet_outputs = cuda_graph.run_with_inputs(hidet_inputs)
    # transfer the hidet tensor to numpy output
    numpy_outputs = [out.numpy() for out in hidet_outputs]
    # consume the output
    output_consumer(numpy_outputs)
t2 = time.time()
outputs_no_pipe = outputs.copy()
outputs = []
print('Run without pipeline: {:.1f} milliseconds'.format((t2 - t1) * 1000))

# %%
# Run with pipeline
# ~~~~~~~~~~~~~~~~~
#
# .. tip::
#   :class: margin
#
#   The :func:`~hidet.runtime.cuda_graph.cuda_graph_pipeline_execute` function use a callback function ``output_consumer``
#   to feed the output data. Another way is to use the :func:`~hidet.runtime.cuda_graph.cuda_graph_pipeline_iterator` generator
#   that returns an iterator to iterate over the output data.
#
# Hidet provides the :func:`~hidet.runtime.cuda_graph.cuda_graph_pipeline_execute` function to
# pipeline the data preprocessing and postprocessing with the model inference.
# It takes the following arguments:
#
# - ``cuda_graph``: the :class:`~hidet.runtime.cuda_graph.CudaGraph` to run.
# - ``input_iterator``: the iterator to get the input data.
# - ``output_consumer``: the function to consume the output data.

t1 = time.time()
cuda_graph_pipeline_execute(cuda_graph, input_iterator(), output_consumer)
t2 = time.time()
print('Run with pipeline: {:.1f} milliseconds'.format((t2 - t1) * 1000))

# %%
# .. note::
#   The latency should be reduced a lot when using pipeline.
#   But it is still slightly larger than the latency of running all samples on GPU.
#   This is because there is some overhead on transferring the data between the CPU and GPU that is hard to be avoided.

# %%
# The outputs should be bit-exact
for i in range(num_samples):
    np.testing.assert_allclose(outputs_no_pipe[i], outputs[i], rtol=0, atol=0)
