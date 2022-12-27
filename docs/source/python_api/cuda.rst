hidet.cuda
=============

Contents
--------

**Device Management**

.. autosummary::
   :nosignatures:

    hidet.cuda.available
    hidet.cuda.device_count
    hidet.cuda.current_device
    hidet.cuda.set_device
    hidet.cuda.properties
    hidet.cuda.compute_capability
    hidet.cuda.synchronize
    hidet.cuda.profiler_start
    hidet.cuda.profiler_stop

**Memory Management**

.. autosummary::
   :nosignatures:

    hidet.cuda.malloc
    hidet.cuda.malloc_async
    hidet.cuda.malloc_host
    hidet.cuda.free
    hidet.cuda.free_async
    hidet.cuda.free_host
    hidet.cuda.memset
    hidet.cuda.memset_async
    hidet.cuda.memcpy
    hidet.cuda.memcpy_async
    hidet.cuda.memory_info

**Stream and Event**

.. autosummary::
   :nosignatures:

    hidet.cuda.Stream
    hidet.cuda.ExternalStream
    hidet.cuda.Event
    hidet.cuda.current_stream
    hidet.cuda.default_stream
    hidet.cuda.stream

**CUDA Graph**

.. autosummary::
   :nosignatures:

    hidet.cuda.graph.CudaGraph

Device Management
-----------------

.. autofunction:: hidet.cuda.available

.. autofunction:: hidet.cuda.device_count

.. autofunction:: hidet.cuda.current_device

.. autofunction:: hidet.cuda.set_device

.. autofunction:: hidet.cuda.properties

.. autofunction:: hidet.cuda.compute_capability

.. autofunction:: hidet.cuda.synchronize

.. autofunction:: hidet.cuda.profiler_start

.. autofunction:: hidet.cuda.profiler_stop


Memory Allocation
-----------------

.. autofunction:: hidet.cuda.malloc

.. autofunction:: hidet.cuda.malloc_async

.. autofunction:: hidet.cuda.malloc_host

.. autofunction:: hidet.cuda.free

.. autofunction:: hidet.cuda.free_async

.. autofunction:: hidet.cuda.free_host

.. autofunction:: hidet.cuda.memset

.. autofunction:: hidet.cuda.memset_async

.. autofunction:: hidet.cuda.memcpy

.. autofunction:: hidet.cuda.memcpy_async

.. autofunction:: hidet.cuda.memory_info


CUDA Stream and Event
---------------------

.. autoclass:: hidet.cuda.Stream
  :members:

.. autoclass:: hidet.cuda.ExternalStream
  :members:

.. autoclass:: hidet.cuda.Event
  :members:

.. autofunction:: hidet.cuda.current_stream

.. autofunction:: hidet.cuda.default_stream

.. autofunction:: hidet.cuda.stream

CUDA Graph
----------

.. autoclass:: hidet.cuda.graph.CudaGraph
  :members:


