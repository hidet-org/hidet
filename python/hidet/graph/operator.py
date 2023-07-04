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
from hidet.ir.type import TensorType, DataType
from hidet.ir.expr import Var, Constant
from hidet.ir.task import Task
from hidet.runtime.compiled_task import CompiledTask
from hidet.graph.tensor import empty, Tensor, SymbolVar
from hidet.ffi.ffi import get_last_error, BackendException
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

        self.outputs = self._run()

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

    def _run(self) -> List[Tensor]:
        from hidet.ir.tools import rewrite, simplify, collect

        if all(t.storage is not None for t in self.inputs) and len(collect(self.task, SymbolVar)) == 0:
            return self.imperative_run(self.inputs)
        else:
            output_types: List[TensorType] = [output_node.type for output_node in self.task.outputs]
            outputs: List[Tensor] = []
            remap: Dict[Var, Constant] = {}
            for i, (a, b) in enumerate(zip(self.task.inputs, self.inputs)):
                for d1, d2 in zip(a.type.shape, b.shape):
                    if isinstance(d1, Var) and not (d1 in remap and isinstance(remap[d1], Constant)):
                        remap[d1] = d2
            for i, output_type in enumerate(output_types):
                shape = [simplify(rewrite(d, remap)) for d in output_type.shape]
                outputs.append(
                    Tensor(shape=shape, dtype=output_type.dtype.name, device=self.device, storage=None, trace=(self, i))
                )
            return outputs

    def get_output(self, idx: int) -> Tensor:
        if self.outputs is None:
            outputs = self._run()
        else:
            outputs = self.outputs
        return outputs[idx]

    def _imperative_run_prepare_outputs(self) -> List[Tensor]:
        from hidet.ir.tools import simplify, collect, rewrite
        from hidet.ffi import runtime_api

        # get the mapping from size var to the actual size
        symbolic_shapes = tuple(tuple(d for d in output.type.shape) for output in self.task.outputs)
        used_symbols: List[SymbolVar] = collect(symbolic_shapes, SymbolVar)
        remap: Dict[SymbolVar, Constant] = {}
        for used_symbol in used_symbols:
            try:
                dtype: DataType = used_symbol.type.as_data_type()
                remap[used_symbol] = dtype(runtime_api.get_symbol_value(used_symbol.name))
            except BackendException as e:
                raise RuntimeError('Failed to get the symbol value of "{}"'.format(used_symbol)) from e
        output_shapes = simplify(rewrite(symbolic_shapes, remap))

        # check if all the output shapes are constant
        for shape in output_shapes:
            for d in shape:
                if not isinstance(d, Constant):
                    raise RuntimeError(
                        'The output shape "{}" of "{}" can not be reduced to a constant'.format(d, self.name)
                    )

        # create the output tensors
        output_dtypes: List[DataType] = [output.type.dtype for output in self.task.outputs]
        outputs: List[Tensor] = [
            empty(shape=shape, dtype=dtype, device=self.device) for shape, dtype in zip(output_shapes, output_dtypes)
        ]

        return outputs

    def imperative_run(self, inputs: List[Tensor]) -> List[Tensor]:
        outputs = self.compiled_task.run_async(inputs)

        status = get_last_error()
        if status is not None:
            msg = 'Kernel for operator {} failed. Error:\n{}'.format(self.name, status)
            raise BackendException(msg)

        return outputs

    def reforward(self, inputs: List[Tensor], update_attributes: Optional[Dict[str, Any]] = None) -> List[Tensor]:
        cls = self.__class__
        attributes = self.attrs.copy()
        if update_attributes is not None:
            attributes.update(update_attributes)
        return cls(*inputs, **attributes).outputs
