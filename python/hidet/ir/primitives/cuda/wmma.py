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
Documentation of wmma and mma instructions in PTX:
https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions
"""
from collections import namedtuple
from typing import List, Optional, Union, Tuple

from hidet.ir.builders import FunctionBuilder
from hidet.ir.expr import Expr
from hidet.ir.expr import Var
from hidet.ir.primitives.cuda.funcs import call_cuda
from hidet.ir.primitives.func import register_primitive_function
from hidet.ir.stmt import AsmStmt
from hidet.ir.type import DataType, PointerType, data_type
from hidet.utils import initialize

dtype_short2long = {'f16': 'float16', 'bf16': 'bfloat16', 'tf32': 'tfloat32', 'f32': 'float32'}
dtype_long2short = {'float16': 'f16', 'bfloat16': 'bf16', 'tfloat32': 'tf32', 'float32': 'f32'}

WmmaConfig = namedtuple(
    'WmmaConfig',
    ['shape', 'a_dtype', 'b_dtype', 'c_dtype', 'a_layout', 'b_layout', 'c_layout', 'a_regs', 'b_regs', 'c_regs'],
)
wmma_configs: List[WmmaConfig] = []


@initialize()
def init_wmma_configs():
    # todo: Add integer, float64, and sub-byte type tensor core wmma instructions when needed.

    #  f16 x f16 => f16 or f32
    for shape in [(16, 16, 16), (8, 32, 16), (32, 8, 16)]:
        a_dtype = 'f16'
        b_dtype = 'f16'
        for c_dtype in ['f16', 'f32']:
            for a_layout in ['row', 'col']:
                for b_layout in ['row', 'col']:
                    for c_layout in ['row', 'col']:
                        a_regs = 8
                        b_regs = 8
                        c_regs = 4 if c_dtype == 'f16' else 8
                        config = WmmaConfig(
                            shape, a_dtype, b_dtype, c_dtype, a_layout, b_layout, c_layout, a_regs, b_regs, c_regs
                        )
                        wmma_configs.append(config)

    # bf16 x bf16 => f32
    for shape in [(16, 16, 16), (8, 32, 16), (32, 8, 16)]:
        a_dtype = 'bf16'
        b_dtype = 'bf16'
        c_dtype = 'f32'
        for a_layout in ['row', 'col']:
            for b_layout in ['row', 'col']:
                for c_layout in ['row', 'col']:
                    m, n, _ = shape
                    regs_map = {16: 4, 8: 2, 32: 8}
                    a_regs = regs_map[m]
                    b_regs = regs_map[n]
                    c_regs = 8
                    config = WmmaConfig(
                        shape, a_dtype, b_dtype, c_dtype, a_layout, b_layout, c_layout, a_regs, b_regs, c_regs
                    )
                    wmma_configs.append(config)

    # tf32 x tf32 => f32
    for shape in [(16, 16, 8)]:
        a_dtype = 'tf32'
        b_dtype = 'tf32'
        c_dtype = 'f32'
        for a_layout in ['row', 'col']:
            for b_layout in ['row', 'col']:
                for c_layout in ['row', 'col']:
                    a_regs = 4
                    b_regs = 4
                    c_regs = 8
                    config = WmmaConfig(
                        shape, a_dtype, b_dtype, c_dtype, a_layout, b_layout, c_layout, a_regs, b_regs, c_regs
                    )
                    wmma_configs.append(config)


@initialize()
def register_wmma_load_instructions():
    WmmaLoadConfig = namedtuple('WmmaLoadConfig', ['matrix', 'layout', 'dtype', 'shape', 'num_regs'])
    configs = set()
    for wmma_config in wmma_configs:
        # pylint: disable=unused-variable
        shape, a_dtype, b_dtype, c_dtype, a_layout, b_layout, c_layout, a_regs, b_regs, c_regs = wmma_config
        configs.add(WmmaLoadConfig('a', a_layout, a_dtype, shape, a_regs))
        configs.add(WmmaLoadConfig('b', b_layout, b_dtype, shape, b_regs))

    for matrix, layout, short_dtype, shape, num_regs in configs:
        inst_name = 'wmma.load.{matrix}.sync.aligned.{layout}.{shape}.{dtype}'.format(
            matrix=matrix, layout=layout, shape='m{}n{}k{}'.format(*shape), dtype=short_dtype
        )
        func_name = 'cuda_' + inst_name.replace('.', '_')
        dtype: DataType = data_type(dtype_short2long[short_dtype])
        with FunctionBuilder(name=func_name, kind='cuda_device') as fb:
            # parameters: dst, src, stride
            dst = Var('dst', PointerType(data_type('uint32')))
            src = Var('src', PointerType(dtype))
            stride = Var('stride', data_type('int32'))
            fb.extend_params([dst, src, stride])

            # body
            assert num_regs > 0
            template_sub_strings = [
                inst_name,
                '{{{}}},'.format(', '.join([f'%{i}' for i in range(num_regs)])),
                '[%{}],'.format(num_regs),
                '%{};'.format(num_regs + 1),
            ]
            fb += AsmStmt(
                template_string=' '.join(template_sub_strings),
                outputs=[('=r', dst[i]) for i in range(num_regs)],
                inputs=[('l', src), ('r', stride)],
                is_volatile=False,
            )
        register_primitive_function(name=func_name, func_or_type=fb.func)


@initialize()
def register_wmma_mma_instructions():
    WmmaMmaConfig = namedtuple(
        'WmmaMmaConfig',
        ['shape', 'a_layout', 'b_layout', 'a_dtype', 'b_dtype', 'c_dtype', 'a_num_regs', 'b_num_regs', 'c_num_regs'],
    )
    configs = set()
    for wmma_config in wmma_configs:
        # pylint: disable=unused-variable
        shape, a_dtype, b_dtype, c_dtype, a_layout, b_layout, c_layout, a_regs, b_regs, c_regs = wmma_config
        configs.add(WmmaMmaConfig(shape, a_layout, b_layout, a_dtype, b_dtype, c_dtype, a_regs, b_regs, c_regs))

    for shape, a_layout, b_layout, a_dtype, b_dtype, c_dtype, a_num_regs, b_num_regs, c_num_regs in configs:
        if a_dtype == 'f16' and b_dtype == 'f16':
            inst_name = 'wmma.mma.sync.aligned.{a_layout}.{b_layout}.{shape}.{d_dtype}.{c_dtype}'.format(
                a_layout=a_layout, b_layout=b_layout, shape='m{}n{}k{}'.format(*shape), d_dtype=c_dtype, c_dtype=c_dtype
            )
        else:
            # pylint: disable=line-too-long
            inst_name = (
                'wmma.mma.sync.aligned.{a_layout}.{b_layout}.{shape}.{d_dtype}.{a_dtype}.{b_dtype}.{c_dtype}'.format(
                    a_layout=a_layout,
                    b_layout=b_layout,
                    shape='m{}n{}k{}'.format(*shape),
                    d_dtype=c_dtype,
                    a_dtype=a_dtype,
                    b_dtype=b_dtype,
                    c_dtype=c_dtype,
                )
            )
        func_name = 'cuda_' + inst_name.replace('.', '_')
        uint32_dtype = data_type('uint32')
        with FunctionBuilder(name=func_name, kind='cuda_device') as fb:
            # parameters: a, b, c
            a = Var('a', PointerType(uint32_dtype))
            b = Var('b', PointerType(uint32_dtype))
            c = Var('c', PointerType(uint32_dtype))
            fb.extend_params([a, b, c])

            # body
            template_sub_strings = [
                inst_name,
                '{{{}}},'.format(', '.join([f'%{i}' for i in range(c_num_regs)])),
                '{{{}}},'.format(', '.join([f'%{i}' for i in range(c_num_regs, c_num_regs + a_num_regs)])),
                '{{{}}},'.format(
                    ', '.join([f'%{i}' for i in range(c_num_regs + a_num_regs, c_num_regs + a_num_regs + b_num_regs)])
                ),
                '{{{}}};'.format(', '.join([f'%{i}' for i in range(c_num_regs)])),
            ]
            template_string = ' '.join(template_sub_strings)
            fb += AsmStmt(
                template_string=template_string,
                outputs=[('+r', c[i]) for i in range(c_num_regs)],
                inputs=[('r', a[i]) for i in range(a_num_regs)] + [('r', b[i]) for i in range(b_num_regs)],
                is_volatile=False,
            )
        register_primitive_function(name=func_name, func_or_type=fb.func)


@initialize()
def register_wmma_store_instructions():
    WmmaStoreConfig = namedtuple('WmmaStoreConfig', ['shape', 'layout', 'dtype', 'num_regs'])
    configs = set()
    for wmma_config in wmma_configs:
        # pylint: disable=unused-variable
        shape, a_dtype, b_dtype, c_dtype, a_layout, b_layout, c_layout, a_regs, b_regs, c_regs = wmma_config
        configs.add(WmmaStoreConfig(shape, c_layout, c_dtype, c_regs))

    for shape, layout, dtype, num_regs in configs:
        inst_name = 'wmma.store.d.sync.aligned.{layout}.{shape}.{dtype}'.format(
            layout=layout, shape='m{}n{}k{}'.format(*shape), dtype=dtype
        )
        func_name = 'cuda_' + inst_name.replace('.', '_')
        dtype = data_type(dtype_short2long[dtype])
        with FunctionBuilder(name=func_name, kind='cuda_device') as fb:
            # parameters: dst, src
            dst = Var('dst', PointerType(dtype))
            src = Var('src', PointerType(data_type('uint32')))
            stride = Var('stride', data_type('int32'))
            fb.extend_params([dst, src, stride])

            # body
            template_sub_strings = [
                inst_name,
                '[%{}],'.format(num_regs),
                '{{{}}},'.format(', '.join([f'%{i}' for i in range(num_regs)])),
                '%{};'.format(num_regs + 1),
            ]
            template_string = ' '.join(template_sub_strings)
            fb += AsmStmt(
                template_string=template_string,
                outputs=[],
                inputs=[('r', src[i]) for i in range(num_regs)] + [('l', dst)] + [('r', stride)],
                is_volatile=False,
            )
        register_primitive_function(name=func_name, func_or_type=fb.func)


def default_stride(matrix: str, layout: str, shape: Tuple[int, int, int]) -> int:
    assert matrix in ['a', 'b', 'c']
    assert layout in ['row', 'col']
    m, n, k = shape
    matrix_shape = {'a': (m, k), 'b': (k, n), 'c': (m, n)}
    a, b = matrix_shape[matrix]
    return b if layout == 'row' else a


def wmma_load_a(config: WmmaConfig, reg_addr: Expr, mem_addr: Expr, stride: Optional[Union[Expr, int]] = None):
    func_name = 'wmma.load.{matrix}.sync.aligned.{layout}.{shape}.{dtype}'.format(
        matrix='a', layout=config.a_layout, shape='m{}n{}k{}'.format(*config.shape), dtype=config.a_dtype
    ).replace('.', '_')
    def_stride = default_stride(matrix='a', layout=config.a_layout, shape=config.shape)
    if stride is None:
        stride = def_stride
    else:
        assert stride % def_stride == 0
    return call_cuda(func_name, args=[reg_addr, mem_addr, stride])


def wmma_load_b(config: WmmaConfig, reg_addr: Expr, mem_addr: Expr, stride: Optional[Union[Expr, int]] = None):
    func_name = 'wmma.load.{matrix}.sync.aligned.{layout}.{shape}.{dtype}'.format(
        matrix='b', layout=config.b_layout, shape='m{}n{}k{}'.format(*config.shape), dtype=config.b_dtype
    ).replace('.', '_')
    def_stride = default_stride(matrix='b', layout=config.b_layout, shape=config.shape)
    if stride is None:
        stride = def_stride
    else:
        assert stride % def_stride == 0
    return call_cuda(func_name, args=[reg_addr, mem_addr, stride])


def wmma_mma(config: WmmaConfig, a_regs_addr: Expr, b_regs_addr: Expr, c_regs_addr: Expr):
    head_part = 'wmma.mma.sync.aligned.{a_layout}.{b_layout}.{shape}'.format(
        a_layout=config.a_layout, b_layout=config.b_layout, shape='m{}n{}k{}'.format(*config.shape)
    )
    if config.a_dtype == 'f16' and config.b_dtype == 'f16':
        type_part = '.{d_dtype}.{c_dtype}'.format(d_dtype=config.c_dtype, c_dtype=config.c_dtype)
    else:
        type_part = '.{d_dtype}.{a_dtype}.{b_dtype}.{c_dtype}'.format(
            d_dtype=config.c_dtype, a_dtype=config.a_dtype, b_dtype=config.b_dtype, c_dtype=config.c_dtype
        )
    func_name = (head_part + type_part).replace('.', '_')
    return call_cuda(func_name, args=[a_regs_addr, b_regs_addr, c_regs_addr])


def wmma_store(config: WmmaConfig, mem_addr: Expr, reg_addr: Expr, stride: Optional[Union[Expr, int]] = None):
    func_name = 'wmma.store.d.sync.aligned.{layout}.{shape}.{dtype}'.format(
        layout=config.c_layout, shape='m{}n{}k{}'.format(*config.shape), dtype=config.c_dtype
    ).replace('.', '_')
    def_stride = default_stride(matrix='c', layout=config.c_layout, shape=config.shape)

    if stride is None:
        stride = def_stride
    else:
        assert stride % def_stride == 0

    return call_cuda(func_name, args=[mem_addr, reg_addr, stride])
