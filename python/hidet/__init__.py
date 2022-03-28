import sys
from . import ir
from . import backend
from . import implement
from . import utils
from . import tasks
from . import tos

from .tos import Tensor, Operator, Module

sys.setrecursionlimit(10000)
