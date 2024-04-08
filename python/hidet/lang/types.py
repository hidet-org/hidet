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
from hidet.ir.dtypes import i8, i16, i32, i64, u8, u16, u32, u64, f16, f32, f64, bf16, tf32
from hidet.ir.dtypes import int8, int16, int32, int64, uint8, uint32, uint64, float16, float32, float64, bfloat16
from hidet.ir.dtypes import tfloat32

from hidet.ir.type import void_p, void, byte_p, tensor_pointer_type

from hidet.lang.constructs.declare import register_tensor, shared_tensor, tensor_pointer, tensor, DeclareScope
