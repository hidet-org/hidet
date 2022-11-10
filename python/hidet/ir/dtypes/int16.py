from typing import Any
import warnings
from hidet.ir.type import DataType


class Int16(DataType):
    def __init__(self):
        super().__init__('int16')

    def short_name(self) -> str:
        return 'i16'

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
        if not -32768 <= value <= 32767:
            raise ValueError('Value {} is out of range for {}.'.format(value, self.name))
        value = int(value)
        return Constant(value, self)

    def one(self):
        return self.constant(1)

    def zero(self):
        return self.constant(0)

    def min_value(self):
        return self.constant(-32768)

    def max_value(self):
        return self.constant(32767)


int16 = Int16()
i16 = int16
