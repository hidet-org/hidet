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
import hidet
from hidet.ir.primitives.debug import printf
from hidet.ir.type import void_p
from hidet.ir.expr import symbol_var


def test_ptr_symbol_var():
    from hidet.lang import attrs

    with hidet.script_module() as script_module:

        @hidet.script
        def launch():
            attrs.func_kind = 'public'
            printf("a = %p\n", symbol_var('ptr', dtype=void_p))
            assert symbol_var('ptr', dtype=void_p) == void_p(1231231)

    func = script_module.build()
    print(func.source())
    """
        DLL void hidet_launch() {
          printf("a = %p\n", ((void*)(get_ptr_symbol_value("ptr"))));
          assert((((void*)(get_ptr_symbol_value("ptr"))) == (void*)1231231));
        }
    """
    hidet.ffi.runtime_api.set_ptr_symbol_value('ptr', 1231231)
    func()
