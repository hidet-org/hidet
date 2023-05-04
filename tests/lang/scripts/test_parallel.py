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
import numpy.testing
import hidet


def test_parallel():
    from hidet.lang import printf, attr, grid, repeat, tensor

    with hidet.script_module() as script_module:

        @hidet.script
        def example():
            attr.func_kind = 'host_kernel'
            a = tensor('global', 'float32', shape=[10])

            for i in grid(10, attrs='p'):  # unroll
                a[i] = i

            for i in grid(10, attrs='p2'):  # unroll explicitly
                a[i] = i

            extent = 10
            for i in grid(extent, attrs='u+'):  # explicit unroll, extent must be a compilation-time constant
                printf("i = %d\n", i)

            for i, j in grid(2, 5, attrs='pp'):  # unroll the first loop while keep the second loop unchanged
                a[i * 5 + j] = i

            b = tensor('global', 'float32', shape=[8, 64])
            for w in range(32):
                for i, j in repeat(2, 8).spatial(4, 8).on(w):
                    b[i, j] = i

                for i, j in repeat(2, 8, attrs='pp').spatial(4, 8).on(w):
                    b[i, j] = i

                for i, j in repeat(2, 8, attrs='p.').spatial(4, 8).on(w):
                    b[i, j] = i

                for i, j in repeat(2, 8, attrs='.p').spatial(4, 8).on(w):
                    b[i, j] = i

    ir_module = script_module.ir_module()
    func = hidet.driver.build_ir_module(ir_module)
    source_code = func.source()
    assert "#pragma omp parallel" in source_code
    return func


def matmul(m_size, n_size, k_size):
    from hidet.lang import grid, attr, f32
    from hidet.lang.mapping import spatial

    with hidet.script_module() as script_module:

        @hidet.script
        def matmul(a: f32[m_size, k_size], b: f32[k_size, n_size], c: f32[m_size, n_size]):
            attr.func_kind = 'host_kernel'
            ij_size = m_size * n_size
            for ij in grid(ij_size, 'p'):
                for i, j in spatial(m_size, n_size).on(ij):
                    c[i, j] = 0.0
                    for k in range(k_size):
                        c[i, j] += a[i, k] * b[k, j]

    ir_module = script_module.ir_module()
    return hidet.driver.build_ir_module(ir_module)


def test_parallel_v2():
    m_size, n_size, k_size = 32, 32, 32
    func = matmul(m_size, n_size, k_size)
    a = hidet.randn((m_size, k_size))
    b = hidet.randn((k_size, n_size))
    c = hidet.empty((m_size, n_size))
    cc = a @ b
    func(a, b, c)
    numpy.testing.assert_allclose(c.numpy(), cc.numpy(), atol=1e-5, rtol=1e-5)


if __name__ == '__main__':
    pytest.main([__file__])
