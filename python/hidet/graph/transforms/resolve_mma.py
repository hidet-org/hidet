from typing import List, Optional
import warnings
from .base import GraphPass, PassContext
from hidet.graph.ir import FlowGraph, Operator, Tensor, GraphRewriter
from hidet.graph.ops.definitions import MatmulOp
from hidet.graph.ops.definitions.matmul.matmul import batched_matmul


class ResolveMmaRewriter(GraphRewriter):
    def visit_Operator(self, op: Operator):
        if isinstance(op, MatmulOp):
            a: Tensor = self(op.inputs[0])
            b: Tensor = self(op.inputs[1])
            mma_type: str = PassContext.current().configs['mma']
            reduce_dtype: Optional[str] = PassContext.current().configs['reduce_precision']
            op_mma, ta, tb, tc = [op.attrs[name] for name in ['mma', 'ta', 'tb', 'tc']]
            if op_mma == 'default':
                if mma_type == 'wmma':
                    mma_dtype = self.get_mma_dtype(a.dtype, b.dtype)
                    mma_acc_dtype = self.get_mma_acc_dtype(mma_dtype, reduce_dtype)
                    mma = 'wmma_{}_{}'.format(mma_dtype, mma_acc_dtype)
                elif mma_type == 'mma':
                    mma_dtype = self.get_mma_dtype(a.dtype, b.dtype)
                    mma_acc_dtype = self.get_mma_acc_dtype(mma_dtype, reduce_dtype)
                    mma = 'mma_{}_{}'.format(mma_dtype, mma_acc_dtype)
                elif mma_type == 'mma_custom':
                    mma = 'mma_custom'
                elif mma_type == 'simt':
                    mma = 'simt'
                elif mma_type.startswith('wmma_'):
                    mma = mma_type
                else:
                    raise ValueError('Can not recognize mma_type {}'.format(mma_type))
            else:
                mma = op_mma
            self.memo[op.outputs[0]] = batched_matmul(a, b, algo=op.attrs['algo'], mma=mma, ta=ta, tb=tb, tc=tc)
        else:
            return GraphRewriter.visit_Operator(self, op)

    @staticmethod
    def get_mma_dtype(a_dtype: str, b_dtype: str):
        from hidet.ir.type import float_dtype_rank
        def max_float_dtype(float_dtypes) -> str:
            return max(float_dtypes, key=lambda dtype: float_dtype_rank[dtype])

        dtype = max_float_dtype([a_dtype, b_dtype])
        if dtype not in ['float16', 'bfloat16', 'float32']:
            raise ValueError('Can not recognize data type {} as input data type of matrix multiplication.'.format(dtype))
        return {
            'float16': 'f16',
            'bfloat16': 'bf16',
            'float32': 'tf32'
        }[dtype]

    @staticmethod
    def get_mma_acc_dtype(mma_dtype: str, acc_dtype: Optional[str]):
        if mma_dtype == 'f16':
            if acc_dtype is None:
                return 'f32'
            elif acc_dtype == 'float16':
                return 'f16'
            elif acc_dtype == 'float32':
                return 'f32'
            else:
                raise ValueError()
        elif mma_dtype == 'bf16':
            if acc_dtype != 'float32' and acc_dtype is not None:
                warnings.warn('bfloat16 only support float32 accumulation in wmma instruction, but got {}. float32 is used.'.format(acc_dtype))
            return 'f32'
        elif mma_dtype == 'tf32':
            if acc_dtype != 'float32' and acc_dtype is not None:
                warnings.warn('tfloat32 only support float32 accumulation in wmma instruction, but got {}. float32 is used.'.format(acc_dtype))
            return 'f32'
        else:
            raise ValueError('Can not recognize mma_dtype {}'.format(mma_dtype))


class ResolveMmaPass(GraphPass):
    def process_graph(self, graph: FlowGraph) -> FlowGraph:
        rewriter = ResolveMmaRewriter()
        return rewriter(graph)


def resolve_mma_pass() -> GraphPass:
    return ResolveMmaPass()
