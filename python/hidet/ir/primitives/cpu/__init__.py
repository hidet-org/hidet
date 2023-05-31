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
from . import math

from .avx import avx_f32x4_broadcast, avx_f32x4_fmadd, avx_f32x4_load, avx_f32x4_store, avx_f32x4_setzero
from .avx import avx_f32x8_broadcast, avx_f32x8_fmadd, avx_f32x8_load, avx_f32x8_store, avx_f32x8_setzero
from .avx import (
    avx_f32x8_unpackhi,
    avx_f32x8_unpacklo,
    avx_f32x8_shuffle,
    avx_f32x8_cast_f32x4,
    avx_f32x8_insert_f32x4,
    avx_f32x8_permute2f32x4,
)
from .avx import avx_free, avx_malloc, x86_memcpy, x86_memset, aligned_alloc
from .avx import avx_f32x8_load_aligned, avx_f32x8_store_aligned
from .avx import avx_f32x4_store_aligned, avx_f32x4_load_aligned
