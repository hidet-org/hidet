from . import definitions

from .definitions.conv2d import conv2d, conv2d_gemm, conv2d_winograd
from .definitions.matmul import matmul, batched_matmul
from .definitions.pool import max_pool2d, avg_pool2d
from .definitions.softmax import softmax
from .definitions.activation import relu, sigmoid
from .definitions.norm import batch_norm_infer, instance_norm
from .definitions.image import resize2d
from .definitions.arithmatic import add, sub, multiply, divide, neg, sqrt, rsqrt, sin, cos, pow, erf, tanh
from .definitions.reduce import reduce_mean
from .definitions.transform import squeeze, unsqueeze, flatten, concat, cast, take, rearrange, strided_slice, reshape, transpose, broadcast, pad

