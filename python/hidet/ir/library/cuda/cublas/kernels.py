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
from hidet.ir.expr import Expr
from hidet.ir.primitives.func import call_primitive_func


def gemm(
    m: Union[Expr, int],
    n: Union[Expr, int],
    k: Union[Expr, int],
    type_a: Union[Expr, int],
    type_b: Union[Expr, int],
    type_c: Union[Expr, int],
    a: Expr,
    b: Expr,
    c: Expr,
    trans_a: Union[Expr, bool],
    trans_b: Union[Expr, bool],
    compute_type: Union[Expr, int],
):
    return call_primitive_func(
        func_name='cublas.gemm', args=[m, n, k, type_a, type_b, type_c, a, b, c, trans_a, trans_b, compute_type]
    )


def bgemm(
    bs: Union[Expr, int],
    m: Union[Expr, int],
    n: Union[Expr, int],
    k: Union[Expr, int],
    type_a: Union[Expr, int],
    type_b: Union[Expr, int],
    type_c: Union[Expr, int],
    a: Expr,
    b: Expr,
    c: Expr,
    trans_a: Union[Expr, bool],
    trans_b: Union[Expr, bool],
    compute_type: Union[Expr, int],
):
    return call_primitive_func(
        func_name='cublas.bgemm', args=[bs, m, n, k, type_a, type_b, type_c, a, b, c, trans_a, trans_b, compute_type]
    )
