from typing import Any
from hidet.ir.type import DataType


class TFloat32(DataType):
    def __init__(self):
        super().__init__('tfloat32')

    def short_name(self) -> str:
        return 'tf32'

    def nbytes(self) -> int:
        return 4

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
        return self.constant(-3.40e38)

    def max_value(self):
        return self.constant(3.40e38)


tfloat32 = TFloat32()
tf32 = tfloat32
