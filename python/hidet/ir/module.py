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
from __future__ import annotations
from typing import Dict
from hidet.ir.node import Node
from hidet.ir.type import FuncType
from hidet.ir.expr import Var
from hidet.ir.func import Function


class IRModule(Node):
    """
    The intermediate representation of tensor programs.

    An IRModule contains one or more functions. It is the basic compilation unit of hidet.
    """

    def __init__(self, functions=None, global_vars=None, namespace='', extern_functions: Dict[str, Var] = None):
        self.functions: Dict[str, Function] = functions if functions else {}
        self.global_vars: Dict[str, Var] = global_vars if global_vars else {}
        self.namespace: str = namespace
        self.extern_functions: Dict[str, Var] = {} if extern_functions is None else extern_functions

        assert all(isinstance(func, Function) for func in self.functions.values()) and all(
            isinstance(var, Var) for var in self.global_vars.values()
        )

    def lookup_var(self, name):
        assert name in self.functions, (name, self.functions.keys())
        if name not in self.global_vars:
            func = self.functions[name]
            if isinstance(func, Function):
                self.global_vars[name] = Var(hint=None, type=FuncType.from_func(func), name=name)
            else:
                raise ValueError()

        return self.global_vars[name]

    def add_function(self, name, func: Function):
        if name in self.functions:
            raise ValueError('Function {} has already existed in module.'.format(name))
        else:
            self.functions[name] = func

    def build(self):
        """
        Build the module.

        Returns
        -------
        ret: hidet.runtime.CompiledModule
            The compiled module.
        """
        import os
        from hidet.drivers import build_ir_module
        from hidet.runtime import load_compiled_module
        from hashlib import sha256

        hash_dir = sha256(str(self).encode()).hexdigest()[:16]
        output_dir = os.path.join('./outs/ir_modules', hash_dir)

        if any(func.kind in ['cuda_kernel', 'cuda_internal'] for func in self.functions.values()):
            target = 'cuda'
        else:
            target = 'cpu'

        build_ir_module(self, output_dir, target=target)
        return load_compiled_module(output_dir)
