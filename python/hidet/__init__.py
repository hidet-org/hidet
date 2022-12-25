"""
Hidet is an open-source DNN inference framework based on compilation.
"""
import sys
from . import option
from . import ir
from . import backend
from . import utils
from . import graph
from . import runtime
from . import driver
from . import logging
from . import cuda

from .ir import Task, save_task, load_task
from .ir.dtypes import int8, int16, int32, int64, uint8, uint16, uint32, uint64, float16, float32, float64, boolean
from .ir.dtypes import bfloat16, tfloat32

from .graph import Tensor, Operator, Module, FlowGraph

from .graph import nn
from .graph import ops
from .graph import empty, randn, zeros, ones, full, randint, symbol, array, from_torch
from .graph import empty_like, randn_like, zeros_like, ones_like, symbol_like, full_like, randint_like
from .graph import trace_from, load_graph, save_graph
from .graph import jit
from .graph import from_dlpack
from .graph import frontend

from .graph.frontend import torch
from .graph.frontend import onnx

from .lang import script, script_module

sys.setrecursionlimit(100000)
