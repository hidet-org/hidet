from . import common
from . import implementer

from . import cuda
from . import cpu

from .implementer import Implementer, implement, register_impl, impl_context
from .resolve.random import random_resolve
