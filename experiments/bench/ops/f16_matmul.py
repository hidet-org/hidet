from hidet.ir.func import IRModule
from hidet.graph import Operator, Tensor
from hidet.graph.ops.definitions.matmul.batch_matmul import BatchMatmulTask
from hidet.graph.ops.definitions.utils import input_like
import hidet


class BatchMatmulF16Task(BatchMatmulTask):
    def implement_cuda(self, working_dir: str) -> IRModule:
        from hidet.lang import attr, float16
        batch_size, m_size, n_size, k_size = (self.attributes['batch_size'],
                                              self.attributes['m_size'],
                                              self.attributes['n_size'],
                                              self.attributes['k_size'])
        with hidet.script_module() as script_module:
            @hidet.script
            def batch_matmul_f16_kernel(
                a: float16[batch_size, m_size, k_size],
                b: float16[batch_size, k_size, n_size],
                c: float16[batch_size, m_size, n_size]
            ):
                attr.func_kind = 'cuda_kernel'
                attr.func_name = 'batch_matmul_f16'


class BatchMatmulF16Op(Operator):
    def __init__(self, a: Tensor, b: Tensor):
        super().__init__(inputs=[a, b], task=BatchMatmulF16Task(input_like(a, 'a'), input_like(b, 'b')))


def batch_matmul_f16(a: Tensor, b: Tensor) -> Tensor:
    if a.dtype != 'float16' or b.dtype != 'float16':
        raise ValueError('BatchMatmulF16Op only support float16, got {} and {}'.format(a.dtype, b.dtype))
    return BatchMatmulF16Op(a, b).get_output(0)

