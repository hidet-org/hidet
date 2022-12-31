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
from .utils import Task, Operator, Tensor, TensorNode, compute, input_like


# todo: add GraphInput and GraphOutput special operators here.


class BarrierTask(Task):
    def __init__(self, x: TensorNode):
        y = compute(name='y', shape=x.const_shape(), fcompute=lambda *indices: x[indices])
        super().__init__(name='barrier', inputs=[x], outputs=[y])


class BarrierOp(Operator):
    def __init__(self, x: Tensor):
        super().__init__(inputs=[x], task=BarrierTask(input_like(x, 'x')))


def barrier(x: Tensor) -> Tensor:
    """
    Barrier operator is an identity operator and return the same tensor as input. During
    graph-level optimizations, this operator prevents the fusion of producer and consumer
    of the input tensor and output tensor, respectively. This operator will be eliminated
    at the end of graph-level optimizations.

    Parameters
    ----------
    x: Tensor
        The input tensor.

    Returns
    -------
    y: Tensor
        The output tensor.
    """
    return BarrierOp(x).get_output(0)
