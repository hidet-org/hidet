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


def test_return_type():
    from hidet.lang import printf, attrs
    from hidet.ir.dtypes import float32x8, float32
    from hidet.ir import primitives
    from hidet.lang import address, cast

    with hidet.script_module() as script_module:

        @hidet.script
        def example() -> float32x8:
            attrs.func_kind = 'cpu_internal'
            return primitives.cpu.avx_f32x8_setzero()

        @hidet.script
        def main():
            attrs.func_kind = 'cpu_kernel'

            a = example()
            a_unpacked = cast(address(a), ~float32)
            for i in range(8):
                printf("%f ", a_unpacked[i])
            printf("\n")

    func = script_module.build()
    func()

    return func


if __name__ == '__main__':
    pytest.main([__file__])
