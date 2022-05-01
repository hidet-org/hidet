from . import cpu
from . import cuda

from .cpu import generic_cpu_schedule
from .cuda import generic_cuda_schedule

from .cuda.softmax import softmax_cuda_schedule
from .cuda.reduce import cuda_schedule_reduce_by_default, cuda_schedule_reduce_by_warp_reduce
