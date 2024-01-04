Function
========

Function kinds
--------------

A function can be one of the following kinds:

- ``public``: a public function can be invoked in python directly
- ``cuda_kernel``: a cuda kernel function
- ``cuda_internal``: a cuda device function that can only be invoked by cuda kernel/device functions
- ``cpu_kernel``: a cpu kernel function
- ``cpu_internal``: a cpu function that will be used by other cpu functions

Only the ``public`` functions will be exposed to python. For the modules that defines a kernel function
(i.e., ``cuda_kernel`` or ``cpu_kernel``), and there is not a ``public`` function named ``launch``, then hidet
will automatically create a ``public`` function named ``launch`` that will launch the kernel function.
