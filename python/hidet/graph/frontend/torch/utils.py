from typing import Tuple, Any, List, Union, Dict
from hidet.graph.tensor import Tensor
from hidet.ir.type import DataType
from hidet.ir import dtypes


def dtype_from_torch(torch_dtype) -> DataType:
    import hidet

    if not hidet.torch.available():
        raise RuntimeError('torch is not available')

    import torch

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
    }
    return mapping[torch_dtype]


def symbol_like_torch(tensor) -> Tensor:
    import hidet
    import torch

    if isinstance(tensor, torch.Tensor):
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
        elif isinstance(obj, (str, int, float)):
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

    def visit_atomic(self, a: Union[str, int, float]):
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
        elif isinstance(obj, (str, int, float)):
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

    def visit_atomic(self, a: Union[str, int, float]):
        return a


def serialize_output(obj: Union[Dict, List, Tuple, Tensor, Any]) -> Tuple[Any, List[Tensor]]:
    return Serializer(obj).serialize()


def deserialize_output(obj: Union[Dict, List, Tuple, Any], tensors) -> Any:
    return Deserializer(obj, tensors).deserialize(obj)
