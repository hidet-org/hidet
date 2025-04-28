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
from typing import Tuple, Any, List, Union, Dict, Optional
from pathlib import Path
import torch
import hidet
from hidet.graph.tensor import Tensor
from hidet.ir.type import DataType
from hidet.ir.expr import Expr, is_true
from hidet.ir import dtypes
from hidet.runtime.device import Device
from hidet.ffi import runtime_api
from hidet.utils import prod
from .availability import available


def tensor_from_torch(tensor) -> Tensor:

    if tensor.requires_grad:
        tensor = tensor.detach()
    return hidet.graph.tensor.from_torch(tensor)


def dtype_from_torch(torch_dtype) -> Optional[DataType]:

    if not available():
        raise RuntimeError('torch is not available')

    if torch_dtype is None:
        return None

    if isinstance(torch_dtype, DataType):
        return torch_dtype

    if isinstance(torch_dtype, str):
        torch_dtype = getattr(torch, torch_dtype)

    mapping = {
        torch.float64: dtypes.float64,
        torch.float32: dtypes.float32,
        torch.float: dtypes.float32,
        torch.bfloat16: dtypes.bfloat16,
        torch.float8_e4m3fn: dtypes.float8_e4m3,
        torch.float8_e5m2: dtypes.float8_e5m2,
        torch.float16: dtypes.float16,
        torch.half: dtypes.float16,
        torch.int64: dtypes.int64,
        torch.int32: dtypes.int32,
        torch.int16: dtypes.int16,
        torch.int8: dtypes.int8,
        torch.uint8: dtypes.uint8,
        torch.uint16: dtypes.uint16,
        torch.uint32: dtypes.uint32,
        torch.uint64: dtypes.uint64,
        torch.bool: dtypes.boolean,
        torch.double: dtypes.float64,
        torch.complex64: dtypes.complex64,
        torch.complex128: dtypes.complex128,
    }
    return mapping[torch_dtype]


def dtype_to_torch(dtype: DataType):

    mapping = {
        dtypes.float64: torch.float64,
        dtypes.float32: torch.float32,
        dtypes.bfloat16: torch.bfloat16,
        dtypes.float8_e4m3: torch.float8_e4m3fn,
        dtypes.float8_e5m2: torch.float8_e5m2,
        dtypes.float16: torch.float16,
        dtypes.int64: torch.int64,
        dtypes.int32: torch.int32,
        dtypes.int16: torch.int16,
        dtypes.int8: torch.int8,
        dtypes.uint8: torch.uint8,
        dtypes.uint16: torch.uint16,
        dtypes.uint32: torch.uint32,
        dtypes.uint64: torch.uint64,
        dtypes.boolean: torch.bool,
    }
    return mapping[dtype]


def is_any_torch_float16(torch_dtype) -> bool:

    return torch_dtype in (torch.float16, torch.bfloat16)


def device_from_torch(torch_device) -> Device:
    """
    Convert a device provided by torch to a hidet device.

    Parameters
    ----------
    torch_device: Union[str, torch.device, Device], optional
        The device to convert. If None, the default device is used.

    Returns
    -------
    ret: Device, optional
        The corresponding hidet device.
    """
    if not available():
        raise RuntimeError('torch is not available')

    if torch_device is None:
        return Device('cpu')

    if isinstance(torch_device, Device):
        return torch_device

    if not isinstance(torch_device, torch.device):
        torch_device = torch.device(torch_device)

    assert isinstance(torch_device, torch.device)

    if torch_device.type == 'cpu':
        return Device('cpu')
    elif torch_device.type == 'cuda':
        if torch.version.hip:
            return Device('hip', torch_device.index)
        elif torch.version.cuda:
            return Device('cuda', torch_device.index)
    raise NotImplementedError(f'unsupported torch device {torch_device}')


def symbol_like_torch(tensor) -> Tensor:
    from torch._subclasses.fake_tensor import FakeTensor

    if isinstance(tensor, FakeTensor):
        symbolic_shape = []
        for s in tensor.shape:
            if isinstance(s, int):
                symbolic_shape.append(s)
            else:
                assert isinstance(s, torch.SymInt)
                expr = s.node.expr
                if expr.is_Integer:
                    i = int(s)
                    symbolic_shape.append(i)
                else:
                    assert expr.is_Symbol
                    name = s.node.expr.name
                    symbolic_shape.append(name)
        return hidet.symbol(shape=symbolic_shape, dtype=dtype_from_torch(tensor.dtype).name, device=tensor.device.type)
    elif isinstance(tensor, torch.Tensor):
        return hidet.symbol(
            shape=list(tensor.shape), dtype=dtype_from_torch(tensor.dtype).name, device=tensor.device.type
        )
    else:
        return hidet.graph.tensor.symbol_like(tensor)


class Placeholder:
    def __init__(self, index):
        self.index = index

    def __str__(self):
        return '<{}>'.format(self.index)


class Symbolholder:
    def __init__(self, name: str):
        self.name = name

    def __str__(self):
        return '<{}>'.format(self.name)


def build_tensor_to_fx_node_map(
    example_inputs: List[torch.Tensor], fx_graph: torch.fx.Graph
) -> Dict[torch.Tensor, Tuple[torch.fx.Node, int]]:
    """
    Build a map from tensor to the corresponding fx node in the given graph. We only have map items for tensors because
    only tensor conversions rely on dynamic shape information from the corresponding fx node.

    Parameters
    ----------
    example_inputs: List[torch.Tensor]
        The example inputs to the graph.
    fx_graph: torch.fx.Graph
        The fx graph to build the map for.

    Returns
    -------
    ret: Dict[torch.Tensor, Tuple[torch.fx.Node, int]]
        A map from tensor to fx node and index.
    """
    tensor_to_node = {}
    for idx, (tensor, node) in enumerate(zip(example_inputs, list(fx_graph.nodes))):
        if node.op == 'placeholder':
            if isinstance(tensor, torch.Tensor):
                tensor_to_node[tensor] = (node, idx)
            elif isinstance(tensor, Tuple):
                for i, t in enumerate(tensor):
                    if isinstance(t, Tuple):
                        raise RuntimeError('Only one level of nested input tuples is supported')
                    if isinstance(t, torch.Tensor):
                        tensor_to_node[t] = (node, i)
    return tensor_to_node


class CompileTimeInputConverter:
    """
    This class accepts example_inputs during compilation and produces:
    1. input_format: similar to the given example_inputs, but replace all torch.Tensor and torch.SymInt to
       Placeholder instances.
    2. converted_input: in this object only tensors and symints are converted to hidet tensors and hidet symbol var
       while the input structure is maintained. This converted_input is used for interpreting the graph
       and producing the output.
    3. flatten_tensors: a list of flattened hidet tensors used for tracing out FlowGraph
    """

    def __init__(self, obj: Any, fx_graph: torch.fx.Graph):
        self.obj = obj
        self.current_index = 0
        self.flatten_tensors = []
        self.tensor_to_fx_node_map = build_tensor_to_fx_node_map(obj, fx_graph)

    def convert(self) -> Tuple[Any, Any, List[Tensor]]:
        input_format, converted_input = self.visit(self.obj)
        return input_format, converted_input, self.flatten_tensors

    def visit(self, obj):
        if isinstance(obj, torch.Tensor):
            return self.visit_tensor(obj)
        elif isinstance(obj, Tuple):
            return self.visit_tuple(obj)
        elif isinstance(obj, List):
            return self.visit_list(obj)
        elif isinstance(obj, torch.SymInt):
            return self.visit_symint(obj)
        elif isinstance(obj, int):
            return self.visit_int(obj)
        else:
            raise RuntimeError('Failed to convert compile time input of type {}'.format(type(obj)))

    def visit_tensor(self, t: torch.Tensor):
        # construct input format
        placeholder = Placeholder(self.current_index)
        self.current_index += 1
        # convert torch tensor to hidet tensor
        # import pdb;pdb.set_trace()
        if hidet.option.internal.is_torch_api_use_example_input_shapes():
            hidet_input = symbol_like_torch(t)
        else:
            assert t in self.tensor_to_fx_node_map
            node, index = self.tensor_to_fx_node_map[t]
            fake_input = node.meta['example_value']
            # if the tensor is within a nested tuple we use
            # the index to get the FakeTensor
            if isinstance(fake_input, Tuple):
                fake_input = fake_input[index]
            hidet_input = symbol_like_torch(fake_input)
        # construct flatten tensor
        self.flatten_tensors.append(hidet_input)
        return placeholder, hidet_input

    def visit_tuple_or_list(self, t: Union[Tuple[Any], List[Any]]):
        container_input_format = []
        container_converted_input = []
        for v in t:
            input_format, converted_input = self.visit(v)
            container_input_format.append(input_format)
            container_converted_input.append(converted_input)
        return container_input_format, container_converted_input

    def visit_tuple(self, t: Tuple[Any]):
        return self.visit_tuple_or_list(t)

    def visit_list(self, t: List[Any]):
        return self.visit_tuple_or_list(t)

    def visit_symint(self, t: torch.SymInt):
        expr = t.node.expr
        assert expr.is_Symbol
        symbolholder = Symbolholder(expr.name)

        hidet_input = hidet.symbol_var(expr.name)
        return symbolholder, hidet_input

    def visit_int(self, t: int):
        return t, t


class RuntimeInputConverter:
    """
    This class is initilized with input_format that was created during compilation time.
    It then takes in a runtime input and converts the input into a flatten list of tensors
    that can be passed to the compiled graph. It also sets the value of the symbolic vars
    on the fly.
    """

    def __init__(self, input_format, obj):
        self.obj = obj
        self.input_format = input_format
        self.flatten_inputs = []

    def convert(self):
        self.visit(self.obj, self.input_format)

    def visit(self, obj, input_format):
        if isinstance(obj, torch.Tensor):
            self.visit_tensor(obj, input_format)
        elif isinstance(obj, torch.SymInt):
            self.visit_symint(obj, input_format)
        elif isinstance(obj, Tuple):
            self.visit_tuple(obj, input_format)
        elif isinstance(obj, List):
            self.visit_list(obj, input_format)
        elif isinstance(obj, int):
            self.visit_int(obj, input_format)
        else:
            raise RuntimeError('Failed to convert runtime input of type {}'.format(type(obj)))

    def visit_tensor(self, t: torch.Tensor, input_format):
        assert isinstance(input_format, Placeholder)
        assert len(self.flatten_inputs) == input_format.index
        self.flatten_inputs.append(t)

    def visit_symint(self, t: torch.SymInt, input_format):
        assert isinstance(input_format, Symbolholder)
        runtime_api.set_symbol_value(input_format.name, int(t))

    def visit_tuple_or_list(self, t: Union[Tuple[Any], List[Any]], input_format):
        assert isinstance(input_format, List)
        for i, v in enumerate(t):
            self.visit(v, input_format[i])

    def visit_tuple(self, t: Tuple[Any], input_format):
        return self.visit_tuple_or_list(t, input_format)

    def visit_list(self, t: List[Any], input_format):
        return self.visit_tuple_or_list(t, input_format)

    def visit_int(self, t: int, input_format):
        assert isinstance(input_format, (Symbolholder, int))
        if isinstance(input_format, Symbolholder):
            runtime_api.set_symbol_value(input_format.name, t)


class Serializer:
    def __init__(self, obj: Any):
        self.obj = obj
        self.current_index = 0
        self.tensors = []

    def serialize(self) -> Tuple[Any, List[Tensor]]:
        result = self.visit(self.obj)
        return result, self.tensors

    def visit(self, obj):
        if isinstance(obj, Tensor):
            return self.visit_tensor(obj)
        elif isinstance(obj, dict):
            return self.visit_dict(obj)
        elif isinstance(obj, list):
            return self.visit_list(obj)
        elif isinstance(obj, tuple):
            return self.visit_tuple(obj)
        elif isinstance(obj, (str, int, float, Expr)):
            return self.visit_atomic(obj)
        else:
            raise RuntimeError('Failed to serialize object of type {}'.format(type(obj)))

    def visit_dict(self, obj: Dict[str, Any]):
        return {self.visit(k): self.visit(v) for k, v in obj.items()}

    def visit_list(self, obj: List[Any]):
        return [self.visit(v) for v in obj]

    def visit_tuple(self, t: Tuple[Any]):
        return tuple(self.visit(v) for v in t)

    def visit_tensor(self, t: Tensor):
        placeholder = Placeholder(self.current_index)
        self.current_index += 1
        self.tensors.append(t)
        return placeholder

    def visit_atomic(self, a: Union[str, int, float, Expr]):
        return a


class Deserializer:
    def __init__(self, obj: Any, tensors):

        self.obj = obj
        self.tensors: List[torch.Tensor] = tensors

    def deserialize(self, obj: Any) -> Any:
        return self.visit(obj)

    def visit(self, obj):
        if isinstance(obj, Placeholder):
            return self.visit_placeholder(obj)
        elif isinstance(obj, dict):
            return self.visit_dict(obj)
        elif isinstance(obj, list):
            return self.visit_list(obj)
        elif isinstance(obj, tuple):
            return self.visit_tuple(obj)
        elif isinstance(obj, (str, int, float, Expr)):
            return self.visit_atomic(obj)
        elif isinstance(obj, Tensor):
            return self.visit_tensor(obj)
        else:
            raise RuntimeError('Failed to serialize object of type {}'.format(type(obj)))

    def visit_dict(self, obj: Dict[str, Any]):
        return {self.visit(k): self.visit(v) for k, v in obj.items()}

    def visit_list(self, obj: List[Any]):
        return [self.visit(v) for v in obj]

    def visit_tuple(self, t: Tuple[Any]):
        return tuple(self.visit(v) for v in t)

    def visit_placeholder(self, p: Placeholder):
        return self.tensors[p.index]

    def visit_tensor(self, t: Tensor):
        raise RuntimeError('Tensors should not be present in the serialized object')

    def visit_atomic(self, a: Union[str, int, float, Expr]):
        if isinstance(a, Expr):
            from hidet.ir.tools import simplify_to_int

            # todo: support other types of symbolic vars
            return simplify_to_int(a, instantiate_symbols=True)
        else:
            return a


def convert_compilation_input(obj, fx_graph) -> Tuple[Any, Any, List[Tensor]]:
    return CompileTimeInputConverter(obj, fx_graph).convert()


def convert_runtime_input(input_format: Any, obj: Union[Dict, List, Tuple, Tensor, Any]) -> List[torch.Tensor]:
    converter = RuntimeInputConverter(input_format, obj)
    converter.convert()
    return converter.flatten_inputs


def serialize_output(obj: Union[Dict, List, Tuple, Tensor, Any]) -> Tuple[Any, List[Tensor]]:
    return Serializer(obj).serialize()


def deserialize_output(obj: Union[Dict, List, Tuple, Any], tensors) -> Any:
    return Deserializer(obj, tensors).deserialize(obj)


def relative_absolute_error(actual, expected) -> float:
    """
    Return :math:`max(|actual - expected| / (|expected| + 1))`, which is the minimum eps satisfy

    :math:`|actual - expected| < eps * |expected| + eps`

    Parameters
    ----------
    actual : torch.Tensor
        The actual value
    expected : torch.Tensor
        The expected value

    Returns
    -------
    ret: float
        The relative error
    """

    actual: torch.Tensor = actual.detach().to(torch.float32)
    expected: torch.Tensor = expected.detach().to(torch.float32)
    return float(torch.max(torch.abs(actual - expected) / (torch.abs(expected) + 1.0)))


def resolve_save_dir_multigraph(save_dir: str) -> str:
    func = resolve_save_dir_multigraph
    if not hasattr(func, 'counter'):
        func.counter = {}
    func.counter[save_dir] = func.counter.get(save_dir, 0) + 1
    return str(Path(save_dir) / "graph_{}".format(func.counter[save_dir]))


def normalize_to_scalar(value: Union[Tensor, Expr, float, int, bool]) -> Union[Expr, int, float, bool]:
    if isinstance(value, Tensor):
        if is_true(prod(value.shape) == 1) and value.storage:
            return value.dtype(value.item())
        else:
            raise RuntimeError(f'Cannot convert tensor {value.signature()} to scalar')
    else:
        return value


def convert_to_scalar_if_possible(x: Union[Tensor, Expr, float, int, bool]) -> Optional[Union[Expr, float, int, bool]]:
    if isinstance(x, Tensor):
        if len(x.shape) == 0 and x.storage:
            return x.item()
        return None
    else:
        return x
