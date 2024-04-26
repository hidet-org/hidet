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
from hidet.graph.tensor import Tensor
from hidet.ir.type import DataType
from hidet.ir.expr import Expr, is_true
from hidet.ir import dtypes
from hidet.runtime.device import Device
from hidet.utils import prod
from .availability import available


def dtype_from_torch(torch_dtype) -> Optional[DataType]:
    if not available():
        raise RuntimeError('torch is not available')

    if torch_dtype is None:
        return None

    if isinstance(torch_dtype, DataType):
        return torch_dtype

    import torch

    if isinstance(torch_dtype, str):
        torch_dtype = getattr(torch, torch_dtype)

    mapping = {
        torch.float64: dtypes.float64,
        torch.float32: dtypes.float32,
        torch.float: dtypes.float32,
        torch.bfloat16: dtypes.bfloat16,
        torch.float16: dtypes.float16,
        torch.half: dtypes.float16,
        torch.int64: dtypes.int64,
        torch.int32: dtypes.int32,
        torch.int16: dtypes.int16,
        torch.int8: dtypes.int8,
        torch.uint8: dtypes.uint8,
        torch.bool: dtypes.boolean,
        torch.double: dtypes.float64,
        torch.complex64: dtypes.complex64,
        torch.complex128: dtypes.complex128,
    }
    return mapping[torch_dtype]


def dtype_to_torch(dtype: DataType):
    import torch

    mapping = {
        dtypes.float64: torch.float64,
        dtypes.float32: torch.float32,
        dtypes.bfloat16: torch.bfloat16,
        dtypes.float16: torch.float16,
        dtypes.int64: torch.int64,
        dtypes.int32: torch.int32,
        dtypes.int16: torch.int16,
        dtypes.int8: torch.int8,
        dtypes.uint8: torch.uint8,
        dtypes.boolean: torch.bool,
    }
    return mapping[dtype]


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

    import torch

    if not isinstance(torch_device, torch.device):
        torch_device = torch.device(torch_device)

    assert isinstance(torch_device, torch.device)

    if torch_device.type == 'cpu':
        return Device('cpu')
    elif torch_device.type == 'cuda':
        return Device('cuda', torch_device.index)
    else:
        raise NotImplementedError(f'unsupported torch device {torch_device}')


def symbol_like_torch(tensor) -> Tensor:
    import hidet
    import torch
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
        import torch

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
    import torch

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
