from . import cpu
from . import cuda

from .cpu import generic_cpu_schedule
from .cuda import generic_cuda_schedule

from .cuda.softmax import softmax_cuda_schedule
