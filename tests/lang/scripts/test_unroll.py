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


def test_unroll():
    from hidet.lang import printf, attr, grid, repeat

    with hidet.script_module() as script_module:
        @hidet.script
        def example():
            attr.func_kind = 'host_kernel'

            for i in grid(10, unroll=True):
                printf("i = %d\n", i)

            for i in grid(10, unroll=2):
                printf("i = %d\n", i)

            for i, j in grid(2, 5, unroll=[True, None]):
                printf("i = %d, j = %d\n", i, j)

            for w in range(32):
                for i, j in repeat(2, 8, unroll=[True, None]).spatial(4, 8).on(w):
                    printf("i = %d, j = %d\n", i, j)

    ir_module = script_module.ir_module()
    func = hidet.driver.build_ir_module(ir_module)
    source_code = func.source()
    assert "#pragma unroll" in source_code


if __name__ == '__main__':
    pytest.main([__file__])
