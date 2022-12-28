from .func import register_primitive_function, is_primitive_function, lookup_primitive_function

# base primitive functions
# pylint: disable=redefined-builtin
from .math import sin, cos, tan, sinh, cosh, tanh, asin, acos, atan, asinh, acosh, atanh, expm1
from .math import max, min, exp, pow, sqrt, rsqrt, erf, ceil, log, log2, log10, log1p, round, floor, trunc
from .math import isfinite, isinf, isnan

# function used to debug
from .debug import printf

# cpu primitive functions
from . import cpu

# cuda primitive functions and variables
from . import cuda
from .cuda import thread_idx, block_idx
from .cuda import syncthreads, syncwarp, lds128, sts128, shfl_sync, shfl_up_sync, shfl_down_sync, shfl_xor_sync
from .cuda import active_mask, set_kernel_max_dynamic_smem_bytes
