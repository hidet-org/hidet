from typing import Any
import warnings
from hidet.ir.type import DataType


class Boolean(DataType):
    def __init__(self):
        super().__init__('bool', 'bool', 1)

    def is_float(self) -> bool:
        return False

    def is_integer(self) -> bool:
        # True for 1, False for 0
        return True

    def is_vector(self) -> bool:
        return False

    def constant(self, value: Any):
        from hidet.ir.expr import Constant

        if isinstance(value, float):
            warnings.warn('Converting float to boolean when creating constant.')
        value = bool(value)
        return Constant(value, self)

    def one(self):
        return self.constant(True)

    def zero(self):
        return self.constant(False)

    def min_value(self):
        raise ValueError('Boolean type has no minimum value.')

    def max_value(self):
        raise ValueError('Boolean type has no maximum value.')


boolean = Boolean()
