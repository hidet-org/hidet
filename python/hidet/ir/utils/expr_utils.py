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
from typing import Union
from hidet.ir.expr import Expr, Constant


def as_expr(e: Union[int, float, bool, Expr]):
    if isinstance(e, Expr):
        return e
    elif isinstance(e, int):
        return Constant(value=e, const_type='int32')
    elif isinstance(e, float):
        return Constant(value=e, const_type='float32')
    elif isinstance(e, bool):
        return Constant(value=e, const_type='bool')
    else:
        raise ValueError('Cannot convert {} to hidet.ir.Expr.'.format(e))
