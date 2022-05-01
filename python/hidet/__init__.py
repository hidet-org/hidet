import sys
from . import ir
from . import backend
from . import utils
from . import tos
from . import runtime
from . import driver

from .ir import Task, save_task, load_task

from .tos import Tensor, Operator, Module, FlowGraph

from .tos import ops
from .tos import empty, randn, zeros, ones, symbol, array, empty_like, randn_like, zeros_like, ones_like, symbol_like
from .tos import space_level, opt_level
from .tos import trace_from, load_graph, save_graph

sys.setrecursionlimit(10000)
