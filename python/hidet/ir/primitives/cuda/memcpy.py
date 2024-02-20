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
from hidet.ir.expr import Expr
from hidet.ir.stmt import BlackBoxStmt


def memcpy_async(dst: Expr, src: Expr, count: Expr, kind: str):
    from hidet.ir.primitives.runtime import get_cuda_stream

    kind_map = {
        'cpu_to_cpu': 'cudaMemcpyHostToHost',
        'cpu_to_cuda': 'cudaMemcpyHostToDevice',
        'cuda_to_cpu': 'cudaMemcpyDeviceToHost',
        'cuda_to_cuda': 'cudaMemcpyDeviceToDevice',
    }

    if kind not in kind_map:
        raise RuntimeError(f'Unsupported transfer from {src} to {dst}, candidate kinds are {list(kind_map.keys())}')
    
    return BlackBoxStmt(
        f'cudaMemcpyAsync({{}}, {{}}, {{}}, {kind_map[kind]}, (cudaStream_t){{}});', dst, src, count, get_cuda_stream()
    )
