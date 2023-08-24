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
# %%
import pytest
from hidet.ir.tools.ir_dumper import astext2, parse
from hidet.ir.expr import symbol_var
from hidet.transforms.unify_global_objects import unify_global_objects_pass
from hidet.transforms.flatten_tensor_slice import flatten_tensor_slice_pass
from hidet.transforms.flatten_tensor_index import flatten_tensor_index_pass
from hidet.transforms.generate_launch_func import generate_launch_func_pass
from hidet.transforms.explicit_unroll import explicit_unroll_pass
from hidet.transforms.import_primitive_functions import import_primitive_functions_pass
from hidet.transforms.simplify_stmt import simplify_stmt_pass
from hidet.transforms.expand_let_expr import expand_let_expr_pass
from hidet.transforms.instantiate_symbols import instantiate_symbols_pass
from hidet.transforms.resolve_generic_primitive_function import resolve_primitive_func_pass
from hidet.transforms.inline_function import inline_function_pass
from hidet.transforms.add_explicit_cast import add_explicit_cast_pass
from hidet.transforms.inline_let_stmt import inline_let_stmt_pass
from hidet.transforms.rule_based_simplifier import rule_based_simplify_pass
from hidet.transforms.normalize_const_tensor import normalize_const_tensor_pass
from hidet.transforms.lower_task_mapping import lower_task_mapping_pass
from hidet.transforms.lower_protect_access import lower_protect_access_pass
from hidet.transforms.declare_to_let import declare_to_let_pass
from hidet.transforms.propagate_launch_bound import propagate_launch_bound_pass
from hidet.transforms.check_launch_configuration import check_launch_configuration_pass
from hidet.transforms.lower_special_cast import lower_special_cast_pass
from hidet.transforms.annotate_header_and_libs import annotate_header_and_libs_pass

# from hidet.graph.ops.softmax import SoftmaxTask
from hidet.graph.ops.matmul.matmul_f16 import MatmulF16Task
from hidet.graph.ops.matmul.batch_matmul import BatchMatmulTask
from hidet.graph.ops.softmax import SoftmaxTask
from hidet.graph.ops.attention.attention import AttnTask
from hidet.graph.ops.utils import input_like, tensor_input

def get_matmul_task():
    s = symbol_var('s')
    a = tensor_input('a', 'float16', [s, 256])
    b = tensor_input('b', 'float16', [256, 512])
    task = MatmulF16Task(a, b)
    mods = task.implement_cuda('.')
    mod = mods[0]
    return mod

def get_bmatmul_task(mma_str='simt'):
    s = symbol_var('s')
    a = tensor_input('a', 'float16', [1, s, 256])
    b = tensor_input('b', 'float16', [1, 256, 256])
    task = BatchMatmulTask(a, b, mma_str)
    mods = task.implement_cuda('.')
    mod = mods[0]
    return mod

def get_softmax_task():
    a = tensor_input('a', 'float16', [1, 256])
    task = SoftmaxTask(a, 1)
    mod = task.implement_cuda('.')
    return mod

def get_attn_task():
    s = symbol_var('s')
    h = symbol_var('h')
    q = tensor_input('q', 'float16', [1, h, s, 64])
    k = tensor_input('k', 'float16', [1, h, s, 64])
    v = tensor_input('v', 'float16', [1, h, s, 64])
    task = AttnTask('attn', q, k, v, False)
    mod = task.implement_cuda('.')
    return mod[0]

def generate_ir_modules():
    transforms = [
        lambda x: x,
        unify_global_objects_pass(),
        generate_launch_func_pass(),
        flatten_tensor_slice_pass(),
        lower_protect_access_pass(),
        lower_task_mapping_pass(),
        normalize_const_tensor_pass(),
        declare_to_let_pass(),
        rule_based_simplify_pass(),
        flatten_tensor_index_pass(),
        lower_special_cast_pass(),
        inline_function_pass(),
        resolve_primitive_func_pass(),
        import_primitive_functions_pass(),
        resolve_primitive_func_pass(),
        import_primitive_functions_pass(),
        propagate_launch_bound_pass(),
        add_explicit_cast_pass(),
        declare_to_let_pass(),
        instantiate_symbols_pass(),
        check_launch_configuration_pass(),
        # simplification
        expand_let_expr_pass(),
        inline_let_stmt_pass(),
        explicit_unroll_pass(),
        rule_based_simplify_pass(),
        inline_let_stmt_pass(),
        simplify_stmt_pass(),
        annotate_header_and_libs_pass(),
    ]
    for mod in [get_matmul_task(), get_bmatmul_task(), get_softmax_task(), get_attn_task()]:
        for t in transforms:
            # if hasattr(t, '__name__'):
            #     print(t.__name__)
            # else:
            #     print(t.__class__.__name__)
            mod = t(mod)
            yield mod

def test_parser():
    for mod in generate_ir_modules():
        text = astext2(mod)
        ir_module = parse(text)
        new_text = astext2(ir_module)

