from typing import List
from hidet.ir.dialects.compute import tensor_input, compute, reduce
from hidet.ir.layout import DataLayout
from hidet.ir.task import Task, Grid
from hidet.ir.type import tensor_type
from hidet.ir.functors import inline_compute


def softmax(data_shape: List[int], axis: int):
    pass


def log_softmax(data_shape, axis):
    pass

#
#  task=CustomOp(tensor_pattern, tag='softmax')
#
#  pattern CustomOpPattern()
#
