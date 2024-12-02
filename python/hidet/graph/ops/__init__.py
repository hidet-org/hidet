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
# pylint: disable=redefined-builtin
from .matmul import batch_matmul, matmul, matmul_x86, matmul_cublas, matmul_nt
from .conv1d import conv1d, conv1d_gemm
from .conv1d_transpose import conv1d_transpose
from .conv2d import conv2d, conv2d_channel_last, conv2d_winograd, conv2d_gemm, conv2d_gemm_fp16
from .conv2d import conv2d_gemm_fp16_channel_last, conv2d_gemm_image_transform
from .conv2d_transpose import conv2d_transpose, conv2d_transpose_gemm
from .conv3d import conv3d, conv3d_gemm
from .conv3d_transpose import conv3d_transpose
from .pool import avg_pool2d, avg_pool3d, adaptive_avg_pool1d, adaptive_avg_pool2d, adaptive_avg_pool3d
from .pool import max_pool2d, max_pool3d, adaptive_max_pool1d, adaptive_max_pool2d, adaptive_max_pool3d
from .activation import relu, leaky_relu, sigmoid, hardsigmoid, clip, relu6, prelu, gelu, silu, hardswish
from .activation import logsigmoid, celu, hardshrink, softplus, softsign, tanhshrink
from .activation import softshrink, softmax, softmin, logsoftmax, hardtanh
from .attention import attention
from .normalize import batch_norm_infer, instance_norm, layer_norm, group_norm, lp_norm
from .image import resize2d
from .create import full, arange, linspace, tri
from .arithmetic import add, subtract, multiply, divide, mod, remainder, negative, positive, square
from .arithmetic import floor, ceil, round, trunc, sqrt, rsqrt, pow, abs
from .arithmetic import reciprocal, exp, expm1, log, log2, log10, log1p, logaddexp, erf
from .arithmetic import bitwise_right_shift, bitwise_left_shift, bitwise_and, bitwise_invert, bitwise_or
from .arithmetic import bitwise_xor, maximum, minimum, clamp
from .arithmetic import isfinite, isinf, isnan, sign, where, set_strided_slice, roll
from .arithmetic import sin, cos, tan, sinh, cosh, tanh, asin, acos, atan, asinh, acosh, atanh, atan2
from .complex import real, imag, conj, make_complex
from .compare import equal, not_equal, less, greater, less_equal, greater_equal
from .compare import logical_not, logical_and, logical_or, logical_xor
from .quant import symmetric_quantize, symmetric_dequantize
from .reduce import mean, sum, var, min, max, std, prod, argmin, argmax, all, any
from .cumulative import cumsum
from .transform import (
    squeeze,
    unsqueeze,
    flatten,
    concat,
    cast,
    take,
    rearrange,
    strided_slice,
    reshape,
    im2col,
    as_strided,
    flip,
)
from .transform import transpose, broadcast, pad, tile, split, conv_pad, expand_dims, gather, index_select, triu, tril
from .transform import permute_dims, meshgrid, repeat_interleave
from .fusion import fused_operator
from .transfer import transfer
from .special import barrier
from .distributed import all_reduce, all_gather, reduce_scatter, wait_tensor
from .linear import einsum
from .embedding_bag import embedding_bag
from .scatter import scatter_add_

from . import utils
