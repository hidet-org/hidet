from typing import List, Optional, Dict, Tuple
from hidet.ir.layout import DataLayout
from hidet.runtime import Storage
from hidet.utils import prod


def convert(v):
    if isinstance(v, float):
        return full(shape=[1], fill_value=v, dtype='float32')
    elif isinstance(v, Tensor):
        return v
    else:
        raise NotImplementedError()


class Tensor:
    def __init__(self,
                 shape: List[int],
                 dtype: str,
                 device: str,
                 storage: Storage,
                 layout: DataLayout = None,
                 trace: Optional[Tuple['Operator', int]] = None):
        self.shape = [int(v) for v in shape]
        self.dtype = str(dtype)
        self.device = device
        self.storage = storage
        self.layout = layout if layout else DataLayout.row_major(shape)
        self.trace = trace

    def __add__(self, other):
        from .ops import add
        return add(self, convert(other))

    def __sub__(self, other):
        from .ops import sub
        return sub(self, convert(other))

    def __mul__(self, other):
        from .ops import multiply
        return multiply(self, convert(other))

    def __truediv__(self, other):
        from .ops import divide
        return divide(self, convert(other))

    def __str__(self):
        return 'Tensor(shape={}, dtype={})'.format(self.shape, self.dtype)

    def zeros_(self):
        raise NotImplementedError()

    def full_(self, fill_value):
        raise NotImplementedError()

    def reshape(self, shape):
        from .ops import reshape
        return reshape(self, shape)

    def flatten(self, start_dim=0, end_dim=-1):
        from .ops import flatten
        return flatten(self, start_dim, end_dim)

    def rsqrt(self):
        from .ops import rsqrt
        return rsqrt(self)


def dtype_bytes(dtype: str):
    bytes_dict = {
        'float32': 4
    }
    return bytes_dict[dtype]


def empty(shape: List[int], dtype: str = 'float32', device: str = 'cuda', layout: Optional[DataLayout] = None) -> Tensor:
    num_bytes = prod(shape) * dtype_bytes(dtype)
    storage = Storage.new(device, num_bytes)
    return Tensor(shape, dtype, device, storage, layout)


def zeros(shape: List[int], dtype: str = 'float32', device: str = 'cuda', layout: Optional[DataLayout] = None) -> Tensor:
    tensor = empty(shape, dtype, device, layout)
    tensor.zeros_()
    return tensor


def full(shape: List[int], fill_value, dtype: str = 'float32', device: str = 'cuda', layout: Optional[DataLayout] = None) -> Tensor:
    tensor = empty(shape, dtype, device, layout)
    tensor.full_(fill_value)
    return tensor
