from typing import Any
import warnings
from hidet.ir.type import DataType


class UInt16(DataType):
    def __init__(self):
        super().__init__('uint16')

    def short_name(self) -> str:
        return 'u16'

    def nbytes(self) -> int:
        return 2

    def is_float(self) -> bool:
        return False

    def is_integer(self) -> bool:
        return True

    def is_vector(self) -> bool:
        return False

    def constant(self, value: Any):
        from hidet.ir.expr import Constant

        if isinstance(value, float):
            warnings.warn('Converting float to int when creating {} constant.'.format(self.name))
        value = int(value)
        if not 0 <= value <= 65535:
            raise ValueError('Value {} is out of range for {}.'.format(value, self.name))
        return Constant(value, self)

    def one(self):
        return self.constant(1)

    def zero(self):
        return self.constant(0)

    def min_value(self):
        return self.constant(0)

    def max_value(self):
        return self.constant(65535)


uint16 = UInt16()
u16 = uint16
