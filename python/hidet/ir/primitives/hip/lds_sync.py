# Licensed under the Apache License,
# Version 2.0 (the "License");
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
from hidet.utils import initialize
from hidet.ir.stmt import AsmStmt
from hidet.ir.builders import FunctionBuilder
from hidet.ir.primitives.func import register_primitive_function
from hidet.ir.primitives.func import call_primitive_func


@initialize()
def register_lds_sync():
    with FunctionBuilder("hip_lds_sync", kind='hip_internal') as fb:
        fb += AsmStmt("s_waitcnt lgkmcnt(0)", is_volatile=True)
        fb += AsmStmt("s_barrier", is_volatile=True)

    register_primitive_function(name="hip_lds_sync", func_or_type=fb.func)


def lds_sync():
    return call_primitive_func("hip_lds_sync", [])
