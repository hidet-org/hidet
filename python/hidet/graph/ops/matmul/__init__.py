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
from .matmul import matmul, MatmulOp, MatmulTask
from .batch_matmul import batch_matmul, BatchMatmulOp, BatchMatmulTask
from . import resolve

from .matmul_f32_x86 import matmul_x86

from .matmul_f32_x86 import MatmulF32Taskx86, Matmulx86Op

from .matmul_f32_x86_refactored import Matmulx86Op_refactored, MatmulF32Taskx86_refactored
from .matmul_f32_x86_refactored import matmul_x86_refactored

