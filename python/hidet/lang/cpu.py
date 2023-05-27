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
# pylint: disable=unused-import

from hidet.ir.primitives.cpu import (
    avx_f32x4_broadcast,
    avx_f32x4_fmadd,
    avx_f32x4_load,
    avx_f32x4_store,
    avx_f32x4_setzero,
)
from hidet.ir.primitives.cpu import (
    avx_f32x8_broadcast,
    avx_f32x8_fmadd,
    avx_f32x8_load,
    avx_f32x8_store,
    avx_f32x8_setzero,
)
from hidet.ir.primitives.cpu import avx_free, avx_malloc, x86_memcpy, x86_memset, aligned_alloc
