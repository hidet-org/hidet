from . import cpu
from . import cuda

from .cuda.softmax import softmax_cuda_schedule
from .cuda.reduce import cuda_schedule_reduce_by_default, cuda_schedule_reduce_by_warp_reduce

from .common import Schedule
