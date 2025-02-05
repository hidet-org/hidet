from typing import List, Union
import pytest
import hidet
from hidet import Tensor
from hidet.graph.ops.opaque import OpaqueOperator
from hidet.ir.dtypes import float32
from hidet.ir import IRModule

hidet.option.cache_dir('./outs/cache')


class OpaqueMatmul(OpaqueOperator):
    def __init__(self, x: Tensor, y: Tensor):
        super().__init__(name='matmul', inputs={'x': x, 'y': y})

    def symbolic_forward(self, x: Tensor, y: Tensor):
        assert x.dtype == y.dtype == float32
        assert x.device.is_cuda()
        m, k = x.shape
        k, n = y.shape
        return {'z': self.symbol(shape=[m, n], dtype=x.dtype, device=x.device)}

    def implement_cuda(self, inputs: List[Tensor], outputs: List[Tensor]) -> Union[IRModule, List[IRModule]]:
        import hidet
        from hidet.lang import attrs
        from hidet.lang.types import f32
        from hidet.lang.cuda import threadIdx, blockIdx

        m_size, k_size = inputs[0].shape
        k_size, n_size = inputs[1].shape

        with hidet.script_module() as script_module:

            @hidet.script
            def matmul(x: f32[m_size, k_size], y: f32[k_size, n_size], z: f32[m_size, n_size]):
                attrs.func_kind = 'cuda_kernel'
                attrs.cuda.block_dim = (32, 32)
                attrs.cuda.grid_dim = ((n_size + 31) // 32, (m_size + 31) // 32)

                i = threadIdx.x + blockIdx.x * 32
                j = threadIdx.y + blockIdx.y * 32
                if i < n_size and j < m_size:
                    z[j, i] = 0.0
                    for k in range(k_size):
                        z[j, i] += x[j, k] * y[k, i]

        return script_module.ir_module()


def opaque_matmul(x: Tensor, y: Tensor) -> Tensor:
    return OpaqueMatmul(x, y).outputs[0]


@pytest.mark.requires_cuda
def test_opaque_operator():
    a = hidet.randn([128, 128], dtype='float32', device='cuda')
    b = hidet.randn([128, 128], dtype='float32', device='cuda')
    c1 = opaque_matmul(a, b)
    c2 = a @ b
    hidet.utils.assert_close(c1, c2)


if __name__ == '__main__':
    pytest.main([__file__])
