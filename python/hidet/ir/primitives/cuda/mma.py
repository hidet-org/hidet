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
"""
Please refer to the following section in PTX manual for the details of MMA instructions:
  https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-for-mma
"""
from typing import List, Dict
from hidet.ir.mapping import TaskMapping, row_spatial, col_spatial, row_repeat, col_repeat
from hidet.utils import initialize
from hidet.ir.type import PointerType, DataType, data_type
from hidet.ir.expr import Var, Expr, cast
from hidet.ir.stmt import AsmStmt, AssignStmt, asm, DeclareStmt
from hidet.ir.func import Function
from hidet.ir.builders import FunctionBuilder
from hidet.ir.primitives.func import register_primitive_function
from hidet.ir.primitives.cuda.funcs import call_cuda


def num_regs(short_dtype: str, num_elements: int) -> int:
    num_bytes = data_type(short_dtype).nbytes * num_elements
    assert num_bytes % (4 * 32) == 0
    return num_bytes // (4 * 32)


class MmaConfig:
    def __init__(self, m, n, k, input_dtype, output_dtype, a_load_map, b_load_map, c_store_map):
        self.m: int = m
        self.n: int = n
        self.k: int = k
        self.input_dtype: str = input_dtype
        self.output_dtype: str = output_dtype
        self.a_load_map: TaskMapping = a_load_map
        self.b_load_map: TaskMapping = b_load_map
        self.c_store_map: TaskMapping = c_store_map
        self.a_elements: int = m * k // 32
        self.b_elements: int = k * n // 32
        self.c_elements: int = m * n // 32
        self.a_regs: int = num_regs(input_dtype, m * k)
        self.b_regs: int = num_regs(input_dtype, k * n)
        self.c_regs: int = num_regs(output_dtype, m * n)

    def inst_name(self) -> str:
        return 'mma.sync.aligned.m{}n{}k{}.row.col.{}.{}.{}.{}'.format(
            self.m, self.n, self.k, self.output_dtype, self.input_dtype, self.input_dtype, self.output_dtype
        )

    @staticmethod
    def m16n8k8_f16_f16():
        return mma_configs['m16n8k8_f16_f16']

    @staticmethod
    def m16n8k8_f16_f32():
        return mma_configs['m16n8k8_f16_f32']

    @staticmethod
    def m16n8k16_f16_f16():
        return mma_configs['m16n8k16_f16_f16']

    @staticmethod
    def m16n8k16_f16_f32():
        return mma_configs['m16n8k16_f16_f32']

    @staticmethod
    def m16n8k8_bf16_f32():
        return mma_configs['m16n8k8_bf16_f32']

    @staticmethod
    def m16n8k16_bf16_f32():
        return mma_configs['m16n8k16_bf16_f32']

    @staticmethod
    def m16n8k4_tf32_f32():
        return mma_configs['m16n8k4_tf32_f32']

    @staticmethod
    def m16n8k8_tf32_f32():
        return mma_configs['m16n8k8_tf32_f32']

    @staticmethod
    def all():
        return list(mma_configs.values())

    def __str__(self):
        return self.inst_name()


mma_configs: Dict[str, MmaConfig] = {}


@initialize()
def register_mma_configs():
    # f16
    for output_dtype in ['f16', 'f32']:
        mma_configs.update(
            {
                'm16n8k8_f16_{}'.format(output_dtype): MmaConfig(
                    m=16,
                    n=8,
                    k=8,
                    input_dtype='f16',
                    output_dtype=output_dtype,
                    a_load_map=row_repeat(2, 1) * row_spatial(8, 4) * row_repeat(1, 2),
                    b_load_map=col_spatial(4, 8) * col_repeat(2, 1),
                    c_store_map=row_repeat(2, 1) * row_spatial(8, 4) * row_repeat(1, 2),
                ),
                'm16n8k16_f16_{}'.format(output_dtype): MmaConfig(
                    m=16,
                    n=8,
                    k=16,
                    input_dtype='f16',
                    output_dtype=output_dtype,
                    a_load_map=col_repeat(2, 2) * row_spatial(8, 4) * row_repeat(1, 2),
                    b_load_map=col_repeat(2, 1) * col_spatial(4, 8) * col_repeat(2, 1),
                    c_store_map=row_repeat(2, 1) * row_spatial(8, 4) * row_repeat(1, 2),
                ),
            }
        )
    # bf16
    mma_configs.update(
        {
            'm16n8k8_bf16_f32': MmaConfig(
                m=16,
                n=8,
                k=8,
                input_dtype='bf16',
                output_dtype='f32',
                a_load_map=row_repeat(2, 1) * row_spatial(8, 4) * row_repeat(1, 2),
                b_load_map=col_spatial(4, 8) * col_repeat(2, 1),
                c_store_map=row_repeat(2, 1) * row_spatial(8, 4) * row_repeat(1, 2),
            ),
            'm16n8k16_bf16_f32': MmaConfig(
                m=16,
                n=8,
                k=16,
                input_dtype='bf16',
                output_dtype='f32',
                a_load_map=col_repeat(2, 2) * row_spatial(8, 4) * row_repeat(1, 2),
                b_load_map=col_repeat(2, 1) * col_spatial(4, 8) * col_repeat(2, 1),
                c_store_map=row_repeat(2, 1) * row_spatial(8, 4) * row_repeat(1, 2),
            ),
        }
    )
    # tf32
    mma_configs.update(
        {
            'm16n8k4_tf32_f32': MmaConfig(
                m=16,
                n=8,
                k=4,
                input_dtype='tf32',
                output_dtype='f32',
                a_load_map=row_repeat(2, 1) * row_spatial(8, 4),
                b_load_map=col_spatial(4, 8),
                c_store_map=row_repeat(2, 1) * row_spatial(8, 4) * row_repeat(1, 2),
            ),
            'm16n8k8_tf32_f32': MmaConfig(
                m=16,
                n=8,
                k=8,
                input_dtype='tf32',
                output_dtype='f32',
                a_load_map=col_repeat(2, 2) * row_spatial(8, 4),
                b_load_map=col_repeat(2, 1) * col_spatial(4, 8),
                c_store_map=row_repeat(2, 1) * row_spatial(8, 4) * row_repeat(1, 2),
            ),
        }
    )


@initialize()
def register_mma_instructions():
    for config in mma_configs.values():
        inst_name = config.inst_name()
        func_name = 'cuda_' + inst_name.replace('.', '_')
        with FunctionBuilder(name=func_name, kind='cuda_device') as fb:
            # parameters: a, b, c
            a = Var('a', PointerType(config.input_dtype))
            b = Var('b', PointerType(config.input_dtype))
            c = Var('c', PointerType(config.output_dtype))
            fb.extend_params([a, b, c])

            # local variables
            ra = Var('ra', PointerType('uint32'))
            rb = Var('rb', PointerType('uint32'))
            rc = Var('rc', PointerType('uint32'))
            fb += DeclareStmt(ra)
            fb += DeclareStmt(rb)
            fb += DeclareStmt(rc)

            # body
            a_regs, b_regs, c_regs = config.a_regs, config.b_regs, config.c_regs
            template_sub_strings = [
                inst_name,
                '{{{}}},'.format(', '.join([f'%{i}' for i in range(c_regs)])),
                '{{{}}},'.format(', '.join([f'%{i}' for i in range(c_regs, c_regs + a_regs)])),
                '{{{}}},'.format(', '.join([f'%{i}' for i in range(c_regs + a_regs, c_regs + a_regs + b_regs)])),
                '{{{}}};'.format(', '.join([f'%{i}' for i in range(c_regs)])),
            ]
            template_string = ' '.join(template_sub_strings)
            fb += AssignStmt(ra, cast(a, ra.type))
            fb += AssignStmt(rb, cast(b, rb.type))
            fb += AssignStmt(rc, cast(c, rc.type))
            fb += AsmStmt(
                template_string=template_string,
                outputs=[('+r', rc[i]) for i in range(c_regs)],
                inputs=[('r', ra[i]) for i in range(a_regs)] + [('r', rb[i]) for i in range(b_regs)],
                is_volatile=False,
            )
        register_primitive_function(name=func_name, func_or_type=fb.func)


def resolve_ldmatrix_func_name(num: int, shared_space_addr: bool = False, trans=False) -> str:
    if num not in [1, 2, 4]:
        raise ValueError('Only support loading 1, 2, or 4 matrices using ldmatrix instruction.')
    return 'ldmatrix_x{num}{shared}{trans}'.format(
        num=num, shared='_shared' if shared_space_addr else '', trans='_trans' if trans else ''
    )


@initialize()
def register_ldmatrix_instructions():
    from hidet.lang import script, u32, void_pointer, attr, ref_u32

    for num in [1, 2, 4]:
        for trans in [False, True]:
            for shared_space_addr in [False, True]:
                func_name = 'cuda_' + resolve_ldmatrix_func_name(
                    num=num, shared_space_addr=shared_space_addr, trans=trans
                )
                inst_name = 'ldmatrix.sync.aligned.m8n8{num}{trans}{ss}.b16'.format(
                    num=f'.x{num}', trans='.trans' if trans else '', ss='.shared' if shared_space_addr else ''
                )
                smem_type = u32 if shared_space_addr else void_pointer
                if num == 1:
                    template = '{inst_name} {{%0}}, [%1];'.format(inst_name=inst_name)

                    @script
                    def cuda_ldmatrix(reg0: ref_u32, smem: smem_type):
                        attr.func_name = func_name
                        attr.func_kind = 'cuda_device'
                        asm(template, outputs=[reg0], inputs=[smem], is_volatile=True)

                    assert isinstance(cuda_ldmatrix, Function)
                    register_primitive_function(cuda_ldmatrix.name, cuda_ldmatrix)

                elif num == 2:
                    template = '{inst_name} {{%0, %1}}, [%2];'.format(inst_name=inst_name)

                    @script
                    def cuda_ldmatrix(reg0: ref_u32, reg1: ref_u32, smem: smem_type):
                        attr.func_name = func_name
                        attr.func_kind = 'cuda_device'
                        asm(template, outputs=[reg0, reg1], inputs=[smem], is_volatile=True)

                    assert isinstance(cuda_ldmatrix, Function)
                    register_primitive_function(cuda_ldmatrix.name, cuda_ldmatrix)
                elif num == 4:
                    template = '{inst_name} {{%0, %1, %2, %3}}, [%4];'.format(inst_name=inst_name)

                    @script
                    def cuda_ldmatrix(reg0: ref_u32, reg1: ref_u32, reg2: ref_u32, reg3: ref_u32, smem: smem_type):
                        attr.func_name = func_name
                        attr.func_kind = 'cuda_device'
                        asm(template, outputs=[reg0, reg1, reg2, reg3], inputs=[smem], is_volatile=True)

                    assert isinstance(cuda_ldmatrix, Function)
                    register_primitive_function(cuda_ldmatrix.name, cuda_ldmatrix)
                else:
                    raise ValueError()


def mma_sync(config: MmaConfig, a_addr: Expr, b_addr: Expr, c_addr: Expr):
    name = config.inst_name().replace('.', '_')
    return call_cuda(func_name=name, args=[a_addr, b_addr, c_addr])


def _print_segment(mapping: TaskMapping, dtype: DataType, addr: Expr, worker_id: Expr, precision: int, msg: str):
    from hidet.ir.dtypes import int32, float32
    from hidet.ir.expr import LogicalAnd, var
    from hidet.ir.primitives import printf, syncwarp
    from hidet.ir.builders import StmtBuilder

    sb = StmtBuilder()
    seg = Var('seg', ~dtype)
    with sb.let(seg, addr):
        if msg:
            with sb.if_then(worker_id == 0):
                sb += printf(f'{msg}\\n')
        with sb.for_loop('i', mapping.task_shape[0], unroll=False) as i:
            with sb.for_loop('j', mapping.task_shape[1], unroll=False) as j:
                p = var('p', int32)
                sb += DeclareStmt(p, int32(0))
                with sb.for_mapping(['ii', 'jj'], mapping, worker_id) as (ii, jj):
                    with sb.if_then(LogicalAnd(ii == i, jj == j)):
                        sb += printf('%.{}f '.format(precision), cast(seg[p], float32))
                    sb += syncwarp()
                    sb += AssignStmt(p, p + 1)
            with sb.if_then(worker_id == 0):
                sb += printf(r'\n')
            sb += syncwarp()
    return sb.finish()


def print_segment_a(config: MmaConfig, a_addr: Expr, worker_id: Expr, precision: int = 2, msg: str = 'Segment A'):
    return _print_segment(config.a_load_map, data_type(config.input_dtype), a_addr, worker_id, precision, msg)


def print_segment_b(config: MmaConfig, b_addr: Expr, worker_id: Expr, precision: int = 2, msg: str = 'Segment B'):
    return _print_segment(config.b_load_map, data_type(config.input_dtype), b_addr, worker_id, precision, msg)


def print_segment_c(config: MmaConfig, c_addr: Expr, worker_id: Expr, precision: int = 2, msg: str = 'Segment C'):
    return _print_segment(config.c_store_map, data_type(config.output_dtype), c_addr, worker_id, precision, msg)


def ldmatrix(regs: List[Expr], smem_addr: Expr, shared_space_addr: bool = False, trans: bool = False):
    """
    Load a matrix 1, 2, or 4 matrix with shape 8x8 from shared memory to registers.

    See Also
    https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-ldmatrix

    Parameters
    ----------
    regs: List[Expr]
        The registers to store the data.
    smem_addr: Expr
        The address of the shared memory.
        If the shared_space_addr is True,
            smem_addr should be the address in shared memory space, and has type of uint32 or int32.
        If the shared_space_addr is False,
            smem_addr should be in the generic memory space, and has type of arbitrary pointer.
        More information about the memory space in cuda programming model can be found in the comment of
        cvta_generic_to_shared primitive function.
    shared_space_addr: bool
        Whether shared memory space is used for smem_addr address.
    trans: bool
        Whether the input matrix in shared memory is transposed.
    """
    num = len(regs)
    func_name = resolve_ldmatrix_func_name(num=num, shared_space_addr=shared_space_addr, trans=trans)
    return call_cuda(func_name, [reg for reg in regs] + [smem_addr])
