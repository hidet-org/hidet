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
from functools import cached_property
import warnings
from hidet.ir.type import DataType


class Boolean(DataType):
    def __init__(self):
        super().__init__('bool', 'bool', 1)

    def is_integer_subbyte(self) -> bool:
        return False

    def is_float(self) -> bool:
        return False

    def is_integer(self) -> bool:
        # True for 1, False for 0
        return True

    def is_complex(self) -> bool:
        return False

    def is_vector(self) -> bool:
        return False

    def constant(self, value: Any):
        from hidet.ir.expr import constant

        if isinstance(value, float):
            warnings.warn('Converting float to boolean when creating constant.')
        value = bool(value)
        return constant(value, self)

    @cached_property
    def one(self):
        return self.constant(True)

    @cached_property
    def zero(self):
        return self.constant(False)

    @cached_property
    def true(self):
        return self.constant(True)

    @cached_property
    def false(self):
        return self.constant(False)

    @property
    def min_value(self):
        raise ValueError('Boolean type has no minimum value.')

    @property
    def max_value(self):
        raise ValueError('Boolean type has no maximum value.')


boolean = Boolean()
