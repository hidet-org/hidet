from typing import Any
from hidet.ir.type import DataType


class Float16(DataType):
    def __init__(self):
        super().__init__('float16')

    def short_name(self) -> str:
        return 'f16'

    def nbytes(self) -> int:
        return 2

    def is_float(self) -> bool:
        return True

    def is_integer(self) -> bool:
        return False

    def is_vector(self) -> bool:
        return False

    def constant(self, value: Any):
        from hidet.ir.expr import Constant

        value = float(value)
        return Constant(value, self)

    def one(self):
        return self.constant(1.0)

    def zero(self):
        return self.constant(0.0)

    def min_value(self):
        return self.constant(-6.55e4)

    def max_value(self):
        return self.constant(6.55e4)


float16 = Float16()
f16 = float16
