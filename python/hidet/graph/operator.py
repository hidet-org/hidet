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
from typing import List, Optional, Dict, Any, Union

from hidet.ir.dtypes import float16, bfloat16, float32
from hidet.ir.task import Task
from hidet.runtime.module import CompiledFunction
from hidet.graph.tensor import empty, empty_like, Tensor
from hidet.ffi.ffi import get_last_error, BackendException
from hidet.runtime.device import Device, instantiate_device


def get_operator_name(op, given_name: Optional[str] = None):
    if given_name is not None:
        return given_name
    cls_name = op.__class__.__name__
    if cls_name.endswith('Op'):
        return cls_name[:-2]
    else:
        return cls_name


class Operator:
    """An operator that takes tensor as input and output."""

    def __init__(
        self,
        inputs: List[Tensor],
        task: Optional[Task],
        outputs: Optional[List[Tensor]] = None,
        name: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ):
        self.inputs: List[Tensor] = inputs
        self.task: Optional[Task] = task
        self.attrs: Dict[str, Any] = attributes if attributes is not None else {}
        self.outputs: Optional[List[Tensor]] = outputs
        self.name: str = get_operator_name(self, name)

        assert all(isinstance(v, Tensor) for v in inputs)

        # cache
        self.task_func: Optional[CompiledFunction] = None

    def __str__(self):
        arguments = ['{}: {}{}'.format(i, t.dtype.name, t.shape) for i, t in enumerate(self.inputs)]
        attributes = ['{}={}'.format(name, str(value)) for name, value in self.attrs.items()]
        return '{}({})'.format(self.name, ', '.join(arguments + attributes))

    @property
    def device(self) -> Device:
        if len(self.inputs) == 0:
            # this is an operator that create a tensor like hidet.full
            # get the device from the operator attributes
            assert 'device' in self.attrs
            return instantiate_device(self.attrs['device'])
        else:
            # when the operator has inputs, get the device from the inputs
            if not all(t.device == self.inputs[0].device for t in self.inputs):
                raise ValueError('All inputs of an operator must be on the same device')
            return self.inputs[0].device

    def run(self) -> List[Tensor]:
        if all(t.storage is not None for t in self.inputs):
            return self.imperative_run(self.inputs)
        else:
            self.outputs = self.symbolic_run()
            return self.outputs

    def get_output(self, idx: int) -> Tensor:
        if self.outputs is None:
            outputs = self.run()
        else:
            outputs = self.outputs
        return outputs[idx]

    def imperative_run(self, inputs: List[Tensor]) -> List[Tensor]:
        self.build_task_func()
        assert len(inputs) + len(self.task.outputs) == len(self.task.parameters)
        output_types = [output.ttype for output in self.task.parameters[-len(self.task.outputs) :]]
        outputs = [
            empty(shape=type.const_shape(), dtype=type.dtype.name, device=self.device, layout=type.layout)
            for type in output_types
        ]

        self.task_func(*inputs, *outputs)

        status = get_last_error()
        if status is not None:
            msg = 'Kernel for operator {} failed. Error:\n{}'.format(self.name, status)
            raise BackendException(msg)

        return outputs

    def symbolic_run(self) -> List[Tensor]:
        output_nodes = self.task.parameters[-len(self.task.outputs) :]
        output_types = [output_node.ttype for output_node in output_nodes]
        outputs = []
        for i, output_type in enumerate(output_types):
            outputs.append(
                Tensor(
                    shape=output_type.const_shape(),
                    dtype=output_type.dtype.name,
                    device=self.device,
                    storage=None,
                    layout=output_type.layout,
                    trace=(self, i),
                )
            )
        return outputs

    def reforward(self, inputs: List[Tensor], update_attributes: Optional[Dict[str, Any]] = None) -> List[Tensor]:
        cls = self.__class__
        if not isinstance(self, Operator) or cls is Operator:
            raise ValueError('Can only reforward operator whose class is a proper class of Operator. Please use .clone')
        attributes = self.attrs.copy()
        if update_attributes is not None:
            attributes.update(update_attributes)
        return cls(*inputs, **attributes).run()

    def clone(self, inputs: List[Tensor], update_attributes: Optional[Dict[str, Any]] = None) -> List[Tensor]:
        cls = self.__class__
        attributes = self.attrs.copy()
        if update_attributes is not None:
            attributes.update(update_attributes)

        new_op = cls.__new__(cls)
        new_op.name = self.name
        new_op.inputs = inputs
        new_op.task = self.task
        new_op.attrs = attributes
        new_op.outputs = new_op.run()
        new_op.task_func = None
        return new_op.outputs

    def dummy_inputs(self) -> List[Tensor]:
        dummy_inputs = []
        for x in self.inputs:
            if x.storage is not None:
                dummy_inputs.append(x)
            else:
                if x.dtype in [float16, bfloat16, float32]:
                    dummy_inputs.append(empty_like(x))
                else:
                    raise ValueError('Can not generate dummy input for dtype {}'.format(x.dtype))
        return dummy_inputs

    def dummy_outputs(self) -> List[Tensor]:
        output_types = [output.ttype for output in self.task.parameters[-len(self.task.outputs) :]]
        dummy_outputs = [
            empty(shape=type.const_shape(), dtype=type.dtype.name, device=self.device, layout=type.layout)
            for type in output_types
        ]
        return dummy_outputs

    def latency(self, warmup=3, number=20, repeat=5, median=True) -> Union[List[float], float]:
        from hidet.testing import benchmark_func

        dummy_inputs = self.dummy_inputs()
        outputs = self.dummy_outputs()
        self.imperative_run(dummy_inputs)
        return benchmark_func(
            lambda: self.task_func(*dummy_inputs, *outputs), warmup=warmup, number=number, repeat=repeat, median=median
        )

    def build_task_func(self):
        from hidet.driver import build_task

        if self.task_func is None:
            self.task_func = build_task(self.task, target_device=self.device.type, load=True)
