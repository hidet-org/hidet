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
from typing import List, Optional, Dict, Any, Union, Tuple
from hidet.ir.type import TensorType, DataType
from hidet.ir.expr import Var, Constant, Expr
from hidet.ir.compute import TensorNode
from hidet.ir.dtypes import float16, bfloat16, float32, int32
from hidet.ir.task import Task
from hidet.runtime.module import CompiledFunction
from hidet.graph.tensor import empty, empty_like, Tensor, SizeVar
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


AttrValue = Union[
    bool,
    int,
    float,
    str,
    None,
    Expr,
    Device,
    'FlowGraph',  # hidet.graph.FlowGraph
    List['AttrValue'],
    Tuple['AttrValue', ...],
    Dict[str, 'AttrValue'],
]


class Operator:
    """An operator that takes tensor as input and output."""

    def __init__(self, inputs: List[Tensor], attributes: Dict[str, AttrValue], task: Optional[Task]):
        assert all(isinstance(v, Tensor) for v in inputs)

        self.name: str = get_operator_name(self)
        self.inputs: List[Tensor] = inputs
        self.attrs: Dict[str, AttrValue] = attributes
        self.task: Optional[Task] = task
        self.outputs: List[Tensor] = []

        # cache
        self._task_func: Optional[CompiledFunction] = None

        self.outputs = self._run()

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

    @property
    def task_func(self) -> CompiledFunction:
        if self._task_func is None:
            self._task_func = self.task.build(target=self.device.type)
        return self._task_func

    def _run(self) -> List[Tensor]:
        from hidet.ir.tools import rewrite, simplify

        if all(t.storage is not None for t in self.inputs) and all(isinstance(t, TensorNode) for t in self.task.params):
            return self.imperative_run(self.inputs, {})
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

    def _imperative_run_prepare_outputs(self, shape_map: Dict[SizeVar, int]) -> List[Tensor]:
        from hidet.ir.tools import simplify

        # get the mapping from size var to the actual size
        symbolic_shapes = tuple(tuple(d for d in output.type.shape) for output in self.task.outputs)
        output_shapes = simplify(symbolic_shapes, {k: int32(v) for k, v in shape_map.items()})

        # check if all the output shapes are constant
        for shape in output_shapes:
            for d in shape:
                if isinstance(d, Constant):
                    raise RuntimeError(
                        'The output shape "{}" of "{}" can not be reduced to a constant'.format(d, self.name)
                    )

        # create the output tensors
        output_dtypes: List[DataType] = [output.type.dtype for output in self.task.outputs]
        outputs: List[Tensor] = [
            empty(shape=shape, dtype=dtype, device=self.device) for shape, dtype in zip(output_shapes, output_dtypes)
        ]

        return outputs

    def imperative_run(self, inputs: List[Tensor], shape_map: Dict[SizeVar, int]) -> List[Tensor]:
        outputs = self._imperative_run_prepare_outputs(shape_map)

        tensor_map = {param: input_tensor for param, input_tensor in zip(self.task.tensor_params, inputs + outputs)}

        args = []
        for param in self.task.params:
            if isinstance(param, SizeVar):
                args.append(shape_map[param])
            elif isinstance(param, TensorNode):
                args.append(tensor_map[param])
            else:
                raise RuntimeError('Unsupported param type: {}'.format(type(param)))

        self.task_func(*args)

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

    def latency(
        self, warmup=3, number=20, repeat=5, median=True, shape_map: Optional[Dict[SizeVar, int]] = None
    ) -> Union[List[float], float]:
        from hidet.testing import benchmark_func

        dummy_inputs = self.dummy_inputs()
        outputs = self.imperative_run(dummy_inputs, {} if shape_map is None else shape_map)
        args = self.task.generate_arguments(dummy_inputs, outputs)
        return benchmark_func(lambda: self.task_func(*args), warmup=warmup, number=number, repeat=repeat, median=median)
