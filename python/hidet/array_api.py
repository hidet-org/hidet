# pylint: disable=unused-import

__array_api_version__ = "2022.12"

# constants
e: float = 2.718281828459045
inf: float = float("inf")
nan: float = float("nan")
pi: float = 3.141592653589793

# creation functions
from .graph.tensor import (
    asarray,
    arange,
    empty,
    empty_like,
    eye,
    from_dlpack,
    full,
    full_like,
    linspace,
    meshgrid,
    ones,
    ones_like,
    zeros,
    zeros_like,
)
from .graph.ops import tril, triu

# data type functions
from .graph.tensor import astype, can_cast, finfo, iinfo, result_type

# dtypes
from .ir.dtypes import (
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
    float16,
    float32,
    float64,
    bfloat16,
    tfloat32,
    boolean as bool,
)

# elementwise functions
from .graph.ops import (
    abs,
    acos,
    acosh,
    add,
    asin,
    asinh,
    atan,
    atan2,
    atanh,
    bitwise_and,
    bitwise_left_shift,
    bitwise_invert,
    bitwise_or,
    bitwise_right_shift,
    bitwise_xor,
    ceil,
    cos,
    cosh,
    divide,
    equal,
    exp,
    expm1,
    floor,
    floor_divide,
    greater,
    greater_equal,
    isfinite,
    isinf,
    isnan,
    less,
    less_equal,
    log,
    log1p,
    log2,
    log10,
    logaddexp,
    logical_and,
    logical_not,
    logical_or,
    logical_xor,
    multiply,
    negative,
    not_equal,
    positive,
    pow,
    remainder,
    round,
    sign,
    sin,
    sinh,
    square,
    sqrt,
    subtract,
    tan,
    tanh,
    trunc,
)

# indexing functions
from .graph.ops import take

# linear algebra functions
# Note: we don't have a linalg module. We just put some linear algebra functions here.
from .graph.ops import matmul

# manipulation functions
from .graph.tensor import broadcast_arrays, broadcast_to
from .graph.ops import concat, expand_dims, flip, permute_dims, reshape, roll, squeeze, stack

# search functions
from .graph.ops import (
    argmax,
    argmin,
    where,
    # hidet currently does not support operators with data-dependent output shape
    # nonzero,
)

# set functions
# Note: not supported yet

# sorting functions
# Note: not supported yet

# statistical functions
from .graph.ops import max, mean, min, prod, std, sum, var

# utility functions
from .graph.ops import all, any
