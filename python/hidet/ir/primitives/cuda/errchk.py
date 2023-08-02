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
from hidet.ir.stmt import BlackBoxStmt


def check_cuda_error():
    stmt = BlackBoxStmt(
        r'''{cudaError_t err = cudaGetLastError(); if (err != cudaSuccess) LOG(ERROR) << "CUDA error: " << '''
        r'''cudaGetErrorString(err) << "\n";}'''
    )
    return stmt
