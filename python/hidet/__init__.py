import sys
from . import ir
from . import backend
from . import utils
from . import graph
from . import runtime
from . import driver

from .ir import Task, save_task, load_task

from .graph import Tensor, Operator, Module, FlowGraph

from .graph.frontend import torch

from .graph import ops
from .graph import empty, randn, zeros, ones, full, randint, symbol, array, empty_like, randn_like, zeros_like, ones_like, symbol_like, full_like, randint_like, from_torch
from .graph import space_level, get_space_level
from .graph import trace_from, load_graph, save_graph
from .graph import jit

from .utils import hidet_set_cache_root as set_cache_root

from .lang import script, script_module

sys.setrecursionlimit(10000)
