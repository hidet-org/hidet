from typing import List, Any, Optional
import torch
from hidet.graph.tensor import Tensor, from_dlpack


def tensor_pad(
    data: List[List[Any]],
    max_length: Optional[int] = None,
    pad_value: Any = 0,
    dtype: str = 'int32',
    device: str = 'cuda',
) -> Tensor:
    if max_length is None:
        max_length = max(len(row) for row in data)
    data = [row + [pad_value] * (max_length - len(row)) for row in data]
    return from_dlpack(torch.tensor(data, dtype=getattr(torch, dtype), device=device))


def tensor(data: Any, dtype: str = 'int32', device: str = 'cuda') -> Tensor:
    return from_dlpack(torch.tensor(data, dtype=getattr(torch, dtype), device=device))
