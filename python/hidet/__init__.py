import sys
from . import ir
from . import backend
from . import implement
from . import utils
from . import tasks
from . import tos

from .tos import Tensor, Operator, Module, FlowGraph

from .tos import empty, randn, zeros, ones, symbol, array
from .tos import lazy_mode, imperative_mode, space_level, opt_level
from .tos import trace_from

sys.setrecursionlimit(10000)
