CPU Specifics
=============

Primitive functions
-------------------

Hidet provides primitives to use the avx instructions in modern cpu. They includes

- ``avx_f32x4_load(...)``: vectorized load 4 f32 values from memory
- ``avx_f32x4_store(...)``: vectorized store 4 f32 values to memory
- ``avx_f32x4_fmadd(...)``: vectorized fused multiply-add operation
- ``avx_f32x4_setzero(...)``: get the zero initialized vector
- ``avx_f32x4_broadcast(...)``: broadcast a scalar to a vector

There are also corresponding ``f32x8`` primitives.

Multi-threading
---------------

Hidet relies on the OpenMP to support multi-threading. To use the multi-threading, please specify the
``p`` attribute of the ``hidet.lang.grid`` or ``hidet.lang.mapping.repeat`` functions.
