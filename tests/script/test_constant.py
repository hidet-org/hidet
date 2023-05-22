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
import pytest
import hidet


def test_const_pointer():
    from hidet.lang import attrs, void_p, printf, int32

    with hidet.script_module() as script_module:

        @hidet.script
        def func():
            attrs.func_kind = 'host_kernel'

            v = int32(0)  # int32 v = 0;
            p_int32 = ~v  # int32* p_int32 = &v;
            p_void = void_p(p_int32)  # void* p_void = (void*)p_int32;
            printf("%p\n", p_void)

            p_void_0 = void_p(0)  # void* p_void_0 = (void*)0;
            printf("%p\n", p_void_0)

            p_void_1 = p_void_0 + 1
            p_void_2 = p_void_1 - 1
            printf("%p\n", p_void_1)
            printf("%p\n", p_void_2)
            printf("%d\n", p_void_1 == p_void_2)
            printf("%d\n", p_void_1 == p_void_2 + 1)
            printf("%p\n", p_int32 + 1)

    func = script_module.build()
    func()


if __name__ == '__main__':
    pytest.main([__file__])
