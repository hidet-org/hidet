from typing import Any
import warnings
from hidet.ir.type import DataType


class UInt32(DataType):
    def __init__(self):
        super().__init__('uint32')

    def short_name(self) -> str:
        return 'u32'

    def nbytes(self) -> int:
        return 4

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
        if not 0 <= value <= 4294967295:
            raise ValueError('Value {} is out of range for {}.'.format(value, self.name))
        return Constant(value, self)

    def one(self):
        return self.constant(1)

    def zero(self):
        return self.constant(0)

    def min_value(self):
        return self.constant(0)

    def max_value(self):
        return self.constant(4294967295)


uint32 = UInt32()
u32 = uint32
