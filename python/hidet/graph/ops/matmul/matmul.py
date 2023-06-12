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
from ..utils import Task, Operator, Tensor, input_like
from ..utils import TensorInput


class MatmulTask(Task):
    def __init__(self, a: TensorInput, b: TensorInput):
        from hidet.ir.compute import cops

        c = cops.matmul(a, b, allow_1d=True)
        super().__init__(name='matmul', inputs=[a, b], outputs=[c])


class MatmulOp(Operator):
    def __init__(self, a: Tensor, b: Tensor):
        task = MatmulTask(input_like(a, 'a'), input_like(b, 'b'))
        super().__init__(inputs=[a, b], attributes={}, task=task)


def matmul(a: Tensor, b: Tensor) -> Tensor:
    return MatmulOp(a, b).get_output(0)
