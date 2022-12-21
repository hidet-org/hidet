from typing import Any
import numpy as np
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

    @property
    def one(self):
        return self.constant(1.0)

    @property
    def zero(self):
        return self.constant(0.0)

    @property
    def min_value(self):
        return self.constant(self._min_value)

    @property
    def max_value(self):
        return self.constant(self._max_value)


float16 = FloatType('float16', 'f16', 2, np.finfo(np.float16).min, np.finfo(np.float16).max)
float32 = FloatType('float32', 'f32', 4, np.finfo(np.float32).min, np.finfo(np.float32).max)
float64 = FloatType('float64', 'f64', 8, np.finfo(np.float64).min, np.finfo(np.float64).max)
bfloat16 = FloatType('bfloat16', 'bf16', 2, -3.4e38, 3.4e38)
tfloat32 = FloatType('tfloat32', 'tf32', 4, -3.4e38, 3.4e38)

f16 = float16
f32 = float32
f64 = float64
bf16 = bfloat16
tf32 = tfloat32
