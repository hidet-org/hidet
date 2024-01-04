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
import math
import hidet
from hidet.cuda.cublas import cublasComputeType


@pytest.mark.parametrize('m, n, k', [[4, 4, 4], [128, 128, 128], [123, 234, 345]])
@pytest.mark.parametrize(
    'dtype, compute_type, tol',
    [
        (hidet.float16, cublasComputeType.CUBLAS_COMPUTE_16F, 1e-2),
        (hidet.float32, cublasComputeType.CUBLAS_COMPUTE_32F, 1e-5),
        (hidet.float64, cublasComputeType.CUBLAS_COMPUTE_64F, 1e-8),
    ],
)
def test_cublas_gemm(m, n, k, dtype, compute_type, tol):
    a = hidet.randn((m, k), device='cuda', dtype=dtype) / math.sqrt(k)
    b = hidet.randn((k, n), device='cuda', dtype=dtype) / math.sqrt(k)
    c = hidet.empty((m, n), device='cuda', dtype=dtype)
    hidet.cuda.cublas.gemm(m, n, k, a.dtype, b.dtype, c.dtype, a, b, c, False, False, compute_type)
    hidet.utils.assert_close(actual=c, expected=a @ b, rtol=tol, atol=tol)


@pytest.mark.parametrize('bs, m, n, k', [[3, 4, 4, 4], [4, 128, 128, 128], [5, 123, 234, 345]])
@pytest.mark.parametrize(
    'dtype, compute_type, tol',
    [
        (hidet.float16, cublasComputeType.CUBLAS_COMPUTE_16F, 1e-2),
        (hidet.float32, cublasComputeType.CUBLAS_COMPUTE_32F, 1e-5),
        (hidet.float64, cublasComputeType.CUBLAS_COMPUTE_64F, 1e-8),
    ],
)
def test_cublas_strided_gemm(bs, m, n, k, dtype, compute_type, tol):
    a = hidet.randn((bs, m, k), device='cuda', dtype=dtype) / math.sqrt(k)
    b = hidet.randn((bs, k, n), device='cuda', dtype=dtype) / math.sqrt(k)
    c = hidet.empty((bs, m, n), device='cuda', dtype=dtype)
    hidet.cuda.cublas.strided_gemm(
        bs, m, n, k, a.dtype, b.dtype, c.dtype, a, b, c, m * k, k * n, m * n, False, False, compute_type
    )
    hidet.utils.assert_close(actual=c, expected=a @ b, rtol=tol, atol=tol)


def test_cublas_library_gemm():
    from hidet.lang import attrs
    from hidet.lang.cuda import cublas
    from hidet.lang.types import f32, i32

    with hidet.script_module() as script_module:

        @hidet.script
        def launch(m_size: i32, n_size: i32, k_size: i32, a: ~f32, b: ~f32, c: ~f32):
            attrs.func_kind = 'public'

            cublas.gemm(
                m_size,
                n_size,
                k_size,
                cublas.as_type_code(f32),
                cublas.as_type_code(f32),
                cublas.as_type_code(f32),
                a,
                b,
                c,
                False,
                False,
                cublas.cublasComputeType.CUBLAS_COMPUTE_32F,
            )

    func = script_module.build()

    m = 234
    n = 345
    k = 456

    a = hidet.randn((m, k), device='cuda', dtype=hidet.float32) / math.sqrt(k)
    b = hidet.randn((k, n), device='cuda', dtype=hidet.float32) / math.sqrt(k)
    c = hidet.empty((m, n), device='cuda', dtype=hidet.float32)

    func(m, n, k, a, b, c)
    hidet.utils.assert_close(actual=c, expected=a @ b, rtol=1e-5, atol=1e-5)


def test_cublas_library_strided_gemm():
    from hidet.lang import attrs
    from hidet.lang.cuda import cublas
    from hidet.lang.types import f32, i32

    with hidet.script_module() as script_module:

        @hidet.script
        def launch(bs: i32, m_size: i32, n_size: i32, k_size: i32, a: ~f32, b: ~f32, c: ~f32):
            attrs.func_kind = 'public'

            cublas.strided_gemm(
                bs,
                m_size,
                n_size,
                k_size,
                cublas.as_type_code(f32),
                cublas.as_type_code(f32),
                cublas.as_type_code(f32),
                a,
                b,
                c,
                m_size * k_size,
                k_size * n_size,
                m_size * n_size,
                False,
                False,
                cublas.cublasComputeType.CUBLAS_COMPUTE_32F,
            )

    func = script_module.build()

    bs = 3
    m = 234
    n = 345
    k = 456

    a = hidet.randn((bs, m, k), device='cuda', dtype=hidet.float32) / math.sqrt(k)
    b = hidet.randn((bs, k, n), device='cuda', dtype=hidet.float32) / math.sqrt(k)
    c = hidet.empty((bs, m, n), device='cuda', dtype=hidet.float32)

    func(bs, m, n, k, a, b, c)
    hidet.utils.assert_close(actual=c, expected=a @ b, rtol=1e-5, atol=1e-5)


if __name__ == '__main__':
    pytest.main([__file__])
