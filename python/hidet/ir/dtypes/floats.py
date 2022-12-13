from typing import Any
import warnings
from hidet.ir.type import DataType


class FloatType(DataType):
    def __init__(self, name, short_name, nbytes, min_value, max_value):
        super().__init__(name, short_name, nbytes)

        self._min_value: float = min_value
        self._max_value: float = max_value

    def is_float(self) -> bool:
        return True

    def is_integer(self) -> bool:
        return False

    def is_vector(self) -> bool:
        return False

    def constant(self, value: Any):
        from hidet.ir.expr import Constant

        if isinstance(value, Constant):
            value = value.value
        value = float(value)

        if value > self._max_value:
            warnings.warn(
                (
                    'Constant value {} is larger than the maximum value {} of data type {}. '
                    'Truncated to maximum value of {}.'
                ).format(value, self._max_value, self.name, self.name)
            )
            value = self._max_value

        if value < self._min_value:
            warnings.warn(
                (
                    'Constant value {} is smaller than the minimum value {} of data type {}. '
                    'Truncated to minimum value of {}.'
                ).format(value, self._min_value, self.name, self.name)
            )
            value = self._min_value

        return Constant(value, self)

    def one(self):
        return self.constant(1.0)

    def zero(self):
        return self.constant(0.0)

    def min_value(self):
        return self.constant(self._min_value)

    def max_value(self):
        return self.constant(self._max_value)


float16 = FloatType('float16', 'f16', 2, -6.55e4, 6.55e4)
float32 = FloatType('float32', 'f32', 4, -3.40e38, 3.40e38)
float64 = FloatType('float64', 'f64', 8, -1.79e308, 1.79e308)
bfloat16 = FloatType('bfloat16', 'bf16', 2, -3.38e38, 3.38e38)
tfloat32 = FloatType('tfloat32', 'tf32', 4, -3.40e38, 3.40e38)

f16 = float16
f32 = float32
f64 = float64
bf16 = bfloat16
tf32 = tfloat32
