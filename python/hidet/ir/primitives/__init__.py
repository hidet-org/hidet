# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from .func import register_primitive_function, is_primitive_function, lookup_primitive_function

# base primitive functions
# pylint: disable=redefined-builtin
from .math import sin, cos, tan, sinh, cosh, tanh, asin, acos, atan, asinh, acosh, atanh, expm1, abs
from .math import max, min, exp, pow, sqrt, rsqrt, erf, ceil, log, log2, log10, log1p, round, floor, trunc
from .math import isfinite, isinf, isnan

from .complex import real, imag, conj, make_complex

# function used to debug
from .debug import printf

# cpu primitive functions
from . import cpu
from .cpu import avx_f32x4_store, avx_f32x4_broadcast, avx_f32x4_fmadd, avx_f32x4_load, avx_f32x4_setzero
from .cpu import avx_free, avx_malloc

# cuda primitive functions and variables
from . import cuda
from .cuda import threadIdx, blockIdx
from .cuda import syncthreads, syncwarp, lds128, sts128, shfl_sync, shfl_up_sync, shfl_down_sync, shfl_xor_sync
from .cuda import active_mask, set_kernel_max_dynamic_smem_bytes
