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
from . import definitions

from .definitions.conv2d import conv2d, conv2d_winograd, conv2d_gemm, conv2d_gemm_image_transform
from .definitions.conv2d_transpose import conv2d_transpose, conv2d_transpose_gemm
from .definitions.conv3d import conv3d, conv3d_gemm
from .definitions.matmul import batch_matmul, matmul
from .definitions.pool import avg_pool2d, avg_pool3d, adaptive_avg_pool1d, adaptive_avg_pool2d, adaptive_avg_pool3d
from .definitions.pool import max_pool2d, max_pool3d, adaptive_max_pool1d, adaptive_max_pool2d, adaptive_max_pool3d
from .definitions.softmax import softmax
from .definitions.activation import relu, leaky_relu, sigmoid, clip, relu6, prelu, gelu
from .definitions.norm import batch_norm_infer, instance_norm, layer_norm
from .definitions.image import resize2d
from .definitions.create import full, arange, linspace
from .definitions.arithmetic import add, subtract, multiply, divide, mod, remainder, negative, positive, square
from .definitions.arithmetic import floor, ceil, round, trunc, sqrt, rsqrt, pow, abs
from .definitions.arithmetic import reciprocal, exp, expm1, log, log2, log10, log1p, logaddexp, erf
from .definitions.arithmetic import bitwise_right_shift, bitwise_left_shift, bitwise_and, bitwise_invert, bitwise_or
from .definitions.arithmetic import bitwise_xor, maximum, minimum
from .definitions.arithmetic import isfinite, isinf, isnan, sign, where
from .definitions.arithmetic import sin, cos, tan, sinh, cosh, tanh, asin, acos, atan, asinh, acosh, atanh, atan2
from .definitions.compare import equal, not_equal, less, greater, less_equal, greater_equal
from .definitions.compare import logical_not, logical_and, logical_or, logical_xor
from .definitions.reduce import mean, sum, var, min, max, std, prod, argmin, argmax, all, any
from .definitions.cumulative import cumsum
from .definitions.transform import squeeze, unsqueeze, flatten, concat, cast, take, rearrange, strided_slice, reshape
from .definitions.transform import transpose, broadcast, pad, tile, split, conv_pad, expand_dims
from .definitions.transform import permute_dims
from .definitions.special import barrier

from .definitions import utils

from . import schedules
