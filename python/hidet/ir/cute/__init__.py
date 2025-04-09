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

from .int_tuple import (
    repeat_like,
    unflatten,
    signum,
    depth,
    flatten,
    rank,
    product,
    product_each,
    prefix_product,
    size,
    inner_product,
    ceil_div,
    shape_div,
    elem_scale,
    congruent,
    compatible,
    crd2idx,
    compact_row_major,
    compact_col_major,
    idx2crd,
    filter_zeros,
    is_integer,
    is_static,
    slice_,
    has_none,
    is_tuple,
    concat_tuple,
)
from .layout import (
    TiledTensorLayout,
    LayoutBase,
    ComposedTensorLayout,
    TensorLayout,
    filter,
    coalesce,
    composition,
    complement,
    left_inverse,
    right_inverse,
    logical_product,
    logical_divide,
    make_layout,
    canonicalize,
    auto_layout,
    layout_auto,
    is_auto_layout,
    max_common_vector,
    slice_and_offset,
    group,
    codomain_from_shape_and_tv_layout,
    canonical_thread_value_layout,
)
from .layout import CopyAtom, ThrValAtom, MmaAtom, Level
from .swizzle import Swizzle, make_swizzle
from .algorithm import TiledCopy, auto_copy, TiledMma, auto_mma
