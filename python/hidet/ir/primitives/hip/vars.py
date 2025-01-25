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
from collections import namedtuple

from hidet.ir.expr import Var
from hidet.ir.dtypes import int32
from hidet.ir.primitives.vars import register_primitive_variable, lookup_primitive_variable
from hidet.utils.py import initialize


@initialize()
def register_hip_primitive_variables():
    for base in ['hipThreadIdx', 'hipBlockIdx', 'hipBlockDim', 'hipGridDim']:
        for suffix in ['x', 'y', 'z']:
            name = '{}_{}'.format(base, suffix)
            register_primitive_variable(name=name, dtype=int32)


def thread_idx(dim='x') -> Var:
    return lookup_primitive_variable('hipThreadIdx_{}'.format(dim))


def block_idx(dim='x') -> Var:
    return lookup_primitive_variable('hipBlockIdx_{}'.format(dim))


def block_dim(dim='x') -> Var:
    return lookup_primitive_variable('hipBlockDim_{}'.format(dim))


def grid_dim(dim='x') -> Var:
    return lookup_primitive_variable('hipGridDim_{}'.format(dim))


dim3 = namedtuple('dim3', field_names=['x', 'y', 'z'])
threadIdx = dim3(thread_idx('x'), thread_idx('y'), thread_idx('z'))
blockIdx = dim3(block_idx('x'), block_idx('y'), block_idx('z'))
blockDim = dim3(block_dim('x'), block_dim('y'), block_dim('z'))
gridDim = dim3(grid_dim('x'), grid_dim('y'), grid_dim('z'))
