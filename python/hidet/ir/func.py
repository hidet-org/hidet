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
from typing import Dict, List, Union
import string
from hidet.ir.node import Node
from hidet.ir.type import BaseType
from hidet.ir.expr import Var, Call
from hidet.ir.stmt import Stmt, BlackBoxStmt


def check_func_name(name: str):
    if len(name) == 0:
        raise ValueError('Do not allow empty function name.')
    for c in name:
        if not (c in string.ascii_lowercase or c in string.ascii_uppercase or c in string.digits or c in '_'):
            raise ValueError('Cannot use {} in function name'.format(repr(c)))


class Function(Node):
    """
    Valid Attrs:
        'kind': str,
            the kind of this function.
                - 'cuda_internal': this is a cuda device function, can only be called by cuda function
                - 'cuda_kernel': this is a cuda kernel function
                - 'cpu_kernel': this is a cpu kernel function
                - 'cpu_internal': this is a cpu function but not a kernel
                - 'public': this is a packed function that wraps kernel function(s)
        'cuda.grid_dim': Union[int, List[int]]
            the grid dimension in cuda launch configuration
        'cuda.block_dim': Union[int, List[int]]
            the block dimension in cuda launch configuration
        'cuda.dynamic_smem_bytes': int
            the dynamic shared memory in cuda launch configuration
        'cuda.min_blocks': int
            the minimal number of thread blocks in launch bound of cuda kernel function
    """

    def __init__(self, name: str, params, body, ret_type, kind: str, attrs=None):
        check_func_name(name)
        self.name: str = name
        self.kind: str = kind
        assert isinstance(kind, str) and kind in [
            'cuda_kernel',
            'cuda_internal',
            'cpu_kernel',
            'cpu_internal',
            'public',
        ]
        self.params: List[Var] = params
        self.body: Stmt = body
        self.ret_type: BaseType = ret_type
        # self.extern_vars: List[Var] = extern_vars if extern_vars else []
        self.attrs: Dict[str, Union[int, float, str, Node]] = attrs if attrs else {}

    def __call__(self, *args, **kwargs) -> Call:
        raise ValueError('Can only call script function in another script function, or lower it to execute.')

    def get_attr(self, attr_name, default=None, allow_missing=False):
        """
        Get attribute of this function.

        When default is not None or allow_missing is True, this function will return the default value (in case
        default is not None) or None (in case default is None) when the attribute is not found. Otherwise,
        this function will raise a KeyError.

        Parameters
        ----------
        attr_name: str
            The name of attribute

        default: Any, optional
            The default value of attribute

        allow_missing: bool, default False

        Returns
        -------
        attr_value: Any
            The value of attribute
        """
        if attr_name in self.attrs:
            return self.attrs[attr_name]
        if default is not None or allow_missing:
            return default
        else:
            raise KeyError('Attribute {} is not found in function {}'.format(attr_name, self.name))

    def use_distributed(self) -> bool:
        """
        Return true if this function involves any distributed primitives
        """

        def _recursive_find(root: Stmt):
            if isinstance(root, BlackBoxStmt):
                if root.template_string.startswith('nccl'):
                    return True
            for child in dir(root):
                if isinstance(child, Stmt):
                    if _recursive_find(child):
                        return True
            return False

        ret = _recursive_find(self.body)
        return ret
