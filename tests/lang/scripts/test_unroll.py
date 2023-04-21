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

            for i in grid(10, attrs='u'):  # unroll
                printf("i = %d\n", i)

            for i in grid(10, attrs='u+'):  # unroll explicitly
                printf("i = %d\n", i)

            for i in grid(5, attrs='u+'):
                for j in grid(i, attrs='u'):
                    for k in grid(2, attrs='u+'):
                        printf("i = %d, j = %d, k = %d\n", i, j, k)

            extent = 10
            for i in grid(extent, attrs='u+'):  # explicit unroll, extent must be a compilation-time constant
                printf("i = %d\n", i)

            for i, j in grid(2, 5, attrs='u.'):  # unroll the first loop while keep the second loop unchanged
                printf("i = %d, j = %d\n", i, j)

            for w in range(32):
                for i, j in repeat(2, 8).spatial(4, 8).on(w):
                    printf("i = %d, j = %d\n", i, j)

                for i, j in repeat(2, 8, attrs='u.').spatial(4, 8).on(w):
                    printf("i = %d, j = %d\n", i, j)

                for i, j in repeat(2, 8, attrs='u+.').spatial(4, 8).on(w):
                    printf("i = %d, j = %d\n", i, j)

                for i, j in repeat(2, 8, attrs='.u+').spatial(4, 8).on(w):
                    printf("i = %d, j = %d\n", i, j)

    ir_module = script_module.ir_module()
    func = hidet.driver.build_ir_module(ir_module)
    source_code = func.source()
    assert "#pragma unroll" in source_code
    return func


if __name__ == '__main__':
    pytest.main([__file__])
