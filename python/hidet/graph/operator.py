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
from typing import List, Optional, Dict, Any
from hidet.ir.type import TensorType
from hidet.ir.task import Task
from hidet.runtime.compiled_task import CompiledTask
from hidet.graph.tensor import Tensor, SymbolVar
from hidet.runtime.device import Device, instantiate_device


def get_operator_name(op):
    cls_name = op.__class__.__name__
    if cls_name.endswith('Op'):
        return cls_name[:-2]
    else:
        return cls_name


class Operator:
    """An operator that takes tensor as input and output."""

    def __init__(self, inputs: List[Tensor], attributes: Dict[str, Any], task: Optional[Task]):
        assert all(isinstance(v, Tensor) for v in inputs)

        self.name: str = get_operator_name(self)
        self.inputs: List[Tensor] = inputs
        self.attrs: Dict[str, Any] = attributes
        self.task: Optional[Task] = task
        self.outputs: List[Tensor] = []

        # cache
        self._compiled_task: Optional[CompiledTask] = None

        self.outputs = self.run()

    def __str__(self):
        arguments = ['{}: {}{}'.format(i, t.dtype.name, t.shape) for i, t in enumerate(self.inputs)]
        attributes = ['{}={}'.format(name, str(value)) for name, value in self.attrs.items()]
        return '{}({})'.format(self.name, ', '.join(arguments + attributes))

    @property
    def device(self) -> Device:
        """Get the device of the output tensor of this operator.

        Returns
        -------
        ret: Device
            The device of the output tensor of this operator.
        """
        # Some notes about self.device:
        #
        #    Each hidet operator has a device property, which is the device of the output tensor of this operator.
        #    For common operators, the device property is inferred from the device of the input tensors. For these
        #    operators, the device for all input tensors and output tensor must be the same.
        #    There are two exceptions:
        #    1. for the operators that create a tensor (e.g., hidet.full), they do not have input tensors.
        #    2. for the transfer operators (e.g., hidet.ops.transfer), the output device is different from the input's
        #    For these operators, they must explicitly set the 'device' attribute, which is used determine the device
        #    of the output tensor.
        #
        if 'device' in self.attrs:
            # this is an operator that create a tensor like hidet.full, or transfer operator
            # get the device from the operator attributes
            return instantiate_device(self.attrs['device'])
        else:
            if len(self.inputs) == 0:
                raise ValueError('Cannot infer device from an operator with no inputs and "device" attribute')
            # when the operator has inputs, get the device from the inputs
            if not all(t.device.target == self.inputs[0].device.target for t in self.inputs):
                raise ValueError('All inputs of an operator must be on the same device')
            return self.inputs[0].device

    @property
    def build_target(self) -> str:
        """
        Get the build target of this operator.

        Returns
        -------
        ret: str
            The build target of this operator.
        """
        from hidet.graph.ops.transfer import TransferOp

        # Some notes about self.build_target:
        #
        #    Each hidet operator has a build_target property, which is the build target of this operator and determines
        #    the scheduling of task and compilation target of the scheduled tensor program.
        #    For common operators, the build_target property is inferred from the device of the output tensor.
        #    There is one exception:
        #    1. for the transfer operators (e.g., hidet.ops.transfer), the build target is always 'cuda' because the
        #    current transfer operators are always between cpu and cuda. Even the output tensor is on cpu (in this case,
        #    the transfer operator copy a tensor from cuda to cpu), the build target is still 'cuda'.

        if isinstance(self, TransferOp):
            return 'cuda'
        if self.device.kind in ["cuda", "vcuda"]:
            return "cuda"
        elif self.device.kind == "cpu":
            return "cpu"
        else:
            raise NotImplementedError()

    @property
    def compiled_task(self) -> CompiledTask:
        if self._compiled_task is None:
            self._compiled_task = self.task.build(target=self.build_target)
        return self._compiled_task

    def run(self) -> List[Tensor]:
        from hidet.ir.tools import collect

        # we imperatively run the operator if
        # 1. all inputs are concrete tensors (i.e., t.storage is not None)
        # 2. there is no symbol variable in the task
        could_imperative_run = (
            all(t.storage is not None for t in self.inputs) and len(collect(self.task, SymbolVar)) == 0
        )

        if could_imperative_run:
            return self.compiled_task.run_async(self.inputs)
        else:
            return self.symbolic_run()

    def symbolic_run(self) -> List[Tensor]:
        from hidet.ir.tools import simplify

        output_types: List[TensorType] = [output_node.type for output_node in self.task.outputs]
        outputs: List[Tensor] = []
        for i, output_type in enumerate(output_types):
            shape = [simplify(d) for d in output_type.shape]
            outputs.append(
                Tensor(shape=shape, dtype=output_type.dtype.name, device=self.device, storage=None, trace=(self, i))
            )
        return outputs

    def reforward(self, inputs: List[Tensor], update_attributes: Optional[Dict[str, Any]] = None) -> List[Tensor]:
        cls = self.__class__
        attributes = self.attrs.copy()
        if update_attributes is not None:
            attributes.update(update_attributes)
        return cls(*inputs, **attributes).outputs
