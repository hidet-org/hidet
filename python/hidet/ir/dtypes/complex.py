# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any
from hidet.ir.type import DataType
from hidet.ir.dtypes.floats import float32, float64


class ComplexType(DataType):
    def __init__(self, name, short_name, base_dtype: DataType):
        super().__init__(name, short_name, 2 * base_dtype.nbytes)
        self.base_dtype: DataType = base_dtype

    def is_float(self) -> bool:
        return False

    def is_integer(self) -> bool:
        return False

    def is_vector(self) -> bool:
        return False

    def is_complex(self) -> bool:
        return True

    def constant(self, value: Any):
        from hidet.ir.expr import Constant

        if isinstance(value, Constant):
            value = value.value

        if isinstance(value, complex):
            return Constant(value, const_type=self)
        elif isinstance(value, (int, float)):
            return Constant(complex(value, 0.0), const_type=self)
        else:
            raise RuntimeError("Invalid constant value for complex type: {}".format(value))

    @property
    def one(self):
        return self.constant(1.0 + 0.0j)

    @property
    def zero(self):
        return self.constant(0.0 + 0.0j)

    @property
    def min_value(self):
        raise RuntimeError("Complex type has no minimum value")

    @property
    def max_value(self):
        raise RuntimeError("Complex type has no maximum value")


complex64 = ComplexType("complex64", "c64", base_dtype=float32)
complex128 = ComplexType("complex128", "c128", base_dtype=float64)

c64 = complex64
c128 = complex128
