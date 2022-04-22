from typing import List, Optional, Dict, Tuple, Sequence, Union
import ctypes
import numpy as np
from functools import partial
from hidet.ir.layout import DataLayout
from hidet.ir.layout.data_layout import RowMajorLayout, ColumnMajorLayout
from hidet.runtime import Storage
from hidet.utils import prod
from hidet.ffi import cuda_api, cuda_kernels


def convert(v):
    if isinstance(v, (float, int)):
        dtype_map = {
            float: 'float32',
            int: 'int64'
        }
        return full(shape=[1], fill_value=v, dtype=dtype_map[type(v)])
    elif isinstance(v, Tensor):
        return v
    else:
        raise NotImplementedError()


class Tensor:
    def __init__(self,
                 shape: Sequence[int],
                 dtype: str,
                 device: str,
                 storage: Optional[Storage],
                 layout: DataLayout = None,
                 trace: Optional[Tuple['Operator', int]] = None):
        from hidet.tos.operator import Operator
        self.shape = [int(v) for v in shape]
        self.dtype = str(dtype)
        self.device = device
        self.storage = storage
        self.layout = layout if layout else DataLayout.row_major(shape)
        self.trace: Optional[Tuple[Operator, int]] = trace

    def __neg__(self) -> 'Tensor':
        from .operators import neg
        return neg(self)

    def __add__(self, other) -> 'Tensor':
        from .operators import add
        return add(self, convert(other))

    def __sub__(self, other) -> 'Tensor':
        from .operators import sub
        return sub(self, convert(other))

    def __mul__(self, other) -> 'Tensor':
        from .operators import multiply
        return multiply(self, convert(other))

    def __truediv__(self, other) -> 'Tensor':
        from .operators import divide
        return divide(self, convert(other))

    def __str__(self):
        head = self.signature()
        if self.storage:
            array_str = str(self.cpu().numpy())
            return '{}\n{}'.format(head, array_str)
        else:
            return head + ' with empty storage'

    def signature(self) -> str:
        return "Tensor(shape={}, dtype='{}', device='{}')".format(self.shape, self.dtype, self.device)

    @property
    def nbytes(self):
        return prod(self.shape) * dtype_bytes(self.dtype)

    @property
    def op(self):
        return self.trace[0] if self.trace else None

    def contiguous(self):
        if isinstance(self.layout, RowMajorLayout):
            return self
        return self.reshape(self.shape)

    def reshape(self, shape: Sequence[int]):
        from .operators import reshape
        return reshape(self, shape)

    def squeeze(self, dims: Sequence[int]):
        from .operators import squeeze
        return squeeze(self, dims)

    def unsqueeze(self, dims: Sequence[int]):
        from .operators import unsqueeze
        return unsqueeze(self, dims)

    def flatten(self, start_dim=0, end_dim=None):
        from .operators import flatten
        return flatten(self, start_dim, end_dim)

    def rsqrt(self):
        from .operators import rsqrt
        return rsqrt(self)

    def cpu(self):
        if self.device == 'cpu':
            return self
        else:
            if self.trace is None:
                return Tensor(self.shape, self.dtype, 'cpu', self.storage.cpu(), self.layout)
            else:
                # lazy mode related
                raise NotImplementedError()

    def cuda(self):
        if self.device == 'cuda':
            return self
        else:
            if self.trace is None:
                return Tensor(self.shape, self.dtype, 'cuda', self.storage.cuda(), self.layout)
            else:
                # lazy mode related
                raise NotImplementedError()

    def numpy(self) -> np.ndarray:
        if self.device != 'cpu':
            return self.cpu().numpy()
        # convert if this tensor is not in row major layout
        storage = self.contiguous().storage
        array = storage.as_array(num_elements=prod(self.shape), dtype=self.dtype)
        return array.reshape(self.shape)


def dtype_bytes(dtype: str):
    bytes_dict = {
        'float32': 4,
        'int32': 4,
        'int64': 8
    }
    return bytes_dict[dtype]


def empty(shape: Sequence[int], dtype: str = 'float32', device: str = 'cuda', layout: Optional[DataLayout] = None) -> Tensor:
    num_bytes = prod(shape) * dtype_bytes(dtype)
    storage = Storage.new(device, num_bytes)
    return Tensor(shape, dtype, device, storage, layout)


def symbol(shape: Sequence[int], dtype: str = 'float32', device: str = 'cuda', layout: Optional[DataLayout] = None) -> Tensor:
    return Tensor(shape, dtype, device, None, layout)


def zeros(shape: Sequence[int], dtype: str = 'float32', device: str = 'cuda', layout: Optional[DataLayout] = None) -> Tensor:
    tensor = empty(shape, dtype, device, layout)
    cuda_api.memset_async(tensor.storage.addr, tensor.nbytes, value=0)
    return tensor


def ones(shape: Sequence[int], dtype: str = 'float32', device: str = 'cuda', layout: Optional[DataLayout] = None) -> Tensor:
    value = {'float32': 1.0, 'int32': 1, 'int64': 1}[dtype]
    return full(shape, value, dtype, device, layout)


def full(shape: Sequence[int], fill_value, dtype: str = 'float32', device: str = 'cuda', layout: Optional[DataLayout] = None) -> Tensor:
    tensor = empty(shape, dtype, device, layout)
    cuda_kernels.fill_value(tensor.storage.addr, tensor.nbytes, value=fill_value, dtype=dtype)
    return tensor


def randn(shape: Sequence[int], dtype: str = 'float32', mean: float = 0.0, stddev: float = 1.0, device: str = 'cuda', layout: Optional[DataLayout] = None) -> Tensor:
    tensor = empty(shape, dtype, device, layout)
    if dtype == 'float32':
        cuda_api.generate_normal(tensor.storage.addr, num_elements=prod(tensor.shape), mean=mean, stddev=stddev)
    else:
        raise NotImplementedError()
    return tensor


def _tensor_like(constructor, data, shape, dtype, device, layout):
    shape = data.shape if shape is None else shape
    dtype = data.dtype if dtype is None else dtype
    device = data.device if device is None else device
    layout = data.layout if layout is None else layout
    return constructor(shape, dtype, device, layout)


def empty_like(data: Tensor, shape: Optional[Sequence[int]] = None, dtype: Optional[str] = None, device: Optional[str] = None, layout: Optional[DataLayout] = None) -> Tensor:
    return _tensor_like(empty, data, shape, dtype, device, layout)


def symbol_like(data: Tensor, shape: Optional[Sequence[int]] = None, dtype: Optional[str] = None, device: Optional[str] = None, layout: Optional[DataLayout] = None) -> Tensor:
    return _tensor_like(symbol, data, shape, dtype, device, layout)


def zeros_like(data: Tensor, shape: Optional[Sequence[int]] = None, dtype: Optional[str] = None, device: Optional[str] = None, layout: Optional[DataLayout] = None) -> Tensor:
    return _tensor_like(zeros, data, shape, dtype, device, layout)


def ones_like(data: Tensor, shape: Optional[Sequence[int]] = None, dtype: Optional[str] = None, device: Optional[str] = None, layout: Optional[DataLayout] = None) -> Tensor:
    return _tensor_like(ones, data, shape, dtype, device, layout)


def full_like(data: Tensor, fill_value, shape: Optional[Sequence[int]] = None, dtype: Optional[str] = None, device: Optional[str] = None, layout: Optional[DataLayout] = None) -> Tensor:
    return _tensor_like(partial(full, fill_value=fill_value), data, shape, dtype, device, layout)


def randn_like(data: Tensor, shape: Optional[Sequence[int]] = None, dtype: Optional[str] = None, device: Optional[str] = None, layout: Optional[DataLayout] = None) -> Tensor:
    return _tensor_like(randn, data, shape, dtype, device, layout)


def void_pointer_to_uint64(p):
    ret = ctypes.cast(ctypes.addressof(p), ctypes.POINTER(ctypes.c_uint64)).contents
    return ret.value


def from_numpy(array: np.ndarray) -> Tensor:
    dtype_convert = {
        np.dtype(np.float32): 'float32',
        np.dtype(np.int64): 'int64'
    }
    if array.dtype not in dtype_convert:
        raise NotImplementedError("Do not support convert np.ndarray with data type '{}'.".format(array.dtype))
    tensor = empty(shape=array.shape, dtype=dtype_convert[array.dtype], device='cpu')
    cuda_api.memcpy_async(src_addr=void_pointer_to_uint64(array.ctypes.data_as(ctypes.c_void_p)),
                          dst_addr=tensor.storage.addr,
                          num_bytes=tensor.nbytes,
                          kind=cuda_api.HostToHost)
    cuda_api.device_synchronization()
    return tensor


def array(obj: Union[List, Tuple, np.ndarray, Tensor]) -> Tensor:
    if isinstance(obj, np.ndarray):
        return from_numpy(obj)
    elif isinstance(obj, Tensor):
        return obj
    else:
        return from_numpy(np.array(obj))
