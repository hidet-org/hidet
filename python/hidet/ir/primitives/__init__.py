from .func import register_primitive_function, is_primitive_function, get_primitive_function


# existing primitive functions
from .func import syncthreads, lds128, sts128, printf, shfl_sync, shfl_up_sync, shfl_xor_sync, shfl_down_sync, expf, cuda_cos, cuda_exp, cuda_round
from .func import cuda_ceil, cuda_floor
from .func import cuda_min, cuda_max, active_mask, cuda_sqrt, cuda_rsqrt, cuda_pow, cuda_erf, cuda_tanh, cuda_sin
from .vars import thread_idx, block_idx, is_primitive_variable, get_primitive_variable
from .base import is_reserved_name
