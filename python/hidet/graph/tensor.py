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
from __future__ import annotations

from typing import List, Optional, Tuple, Sequence, Union
import warnings

import numpy as np

import hidet.runtime.storage
import hidet.cuda
from hidet.ir import dtypes
from hidet.ir.type import DataType, data_type
from hidet.ir.layout import DataLayout, RowMajorLayout
from hidet.runtime.storage import Storage
from hidet.utils import prod
from hidet.utils.overrides import set_module
from hidet.runtime.device import Device, instantiate_device


@set_module('hidet')
class Tensor:
    """An n-dimension array, could be symbolic or concrete.

    This class defines an n-dimension array.

    Parameters
    ----------
    shape: Sequence[int]
        The shape of the tensor.

    dtype: DataType or str
        The data type of the tensor.

    device: Device or str
        The device of the tensor.

    storage: Storage, optional
        The storage of the tensor. None indicates it is a symbolic tensor.

    layout: DataLayout, optional
        The data layout of the tensor.

    trace: Tuple[Operator, int], optional
        Where this tensor is derived from. A trace = (op, i) indicates that this tensor is the i-th output of the op
        operator.
    """

    def __init__(self, shape, dtype, device, storage, layout=None, trace=None):
        from hidet.graph.operator import Operator

        self._shape: List[int] = [int(v) for v in shape]
        self._dtype: DataType = data_type(dtype)
        self._device: Device = instantiate_device(device)
        self._storage: Optional[Storage] = storage
        self._layout: DataLayout = layout if layout else DataLayout.row_major(shape)
        self._trace: Optional[Tuple[Operator, int]] = trace

    @property
    def shape(self) -> Tuple[int, ...]:
        """
        The shape of the tensor.

        The shape is a tuple of integers indicating the size of the tensor along each dimension.

        Returns
        -------
        shape: Tuple[int, ...]
            The shape of the tensor.
        """
        return tuple(self._shape)

    @property
    def dtype(self) -> DataType:
        """
        The data type of the tensor.

        Returns
        -------
        dtype: DataType
            The data type of the tensor.
        """
        return self._dtype

    @property
    def device(self) -> Device:
        """
        The device of the tensor.

        Returns
        -------
        device: Device
            The device of the tensor.
        """
        return self._device

    @property
    def storage(self) -> Optional[Storage]:
        """
        The storage of the tensor.

        Returns
        -------
        storage: Storage
            The storage of the tensor.
        """
        return self._storage

    @property
    def trace(self):
        """
        The producer and the index of outputs of the producer of this tensor.

        This attribute is used to track how this tensor is computed. None indicates this is a leaf tensor where the
        value will be given by the user. Otherwise, it will be a tuple with (operator, index) where operator is the
        producer of this tensor and index is the index of the output of the operator.

        Returns
        -------
        trace: Tuple[Operator, int]
            The trace of this tensor.
        """
        return self._trace

    @property
    def size(self) -> int:
        """
        The number of elements in the tensor.

        Returns
        -------
        size: int
            The number of elements in the tensor.
        """
        return prod(self._shape)

    @property
    def layout(self) -> DataLayout:
        """
        The data layout of the tensor.

        .. note::

          This attribute is experimental and might change in the future.

        Returns
        -------
        layout: DataLayout
            The data layout of the tensor.
        """
        return self._layout

    @property
    def nbytes(self):
        """The number of bytes of the tensor.

        Returns
        -------
        ret: int
            The number of bytes.
        """
        return prod(self.shape) * self.dtype.nbytes

    @property
    def op(self):
        """The operator that produces this tensor.

        Returns
        -------
        ret: hidet.graph.operator.Operator, optional
            The operator that produces this tensor. None indicates it is not traced.
        """
        return self.trace[0] if self.trace else None

    def __pos__(self):
        return self

    def __neg__(self) -> Tensor:
        from .ops import negative

        return negative(self)

    def __add__(self, other) -> Tensor:
        from .ops import add

        return add(self, other)

    def __sub__(self, other) -> Tensor:
        from .ops import subtract

        return subtract(self, other)

    def __mul__(self, other) -> Tensor:
        from .ops import multiply, utils

        return multiply(self, utils.convert_to_tensor(other, self))

    def __truediv__(self, other) -> Tensor:
        from .ops import divide, utils

        return divide(self, utils.convert_to_tensor(other, self))

    def __mod__(self, other) -> Tensor:
        from .ops import mod, utils

        return mod(self, utils.convert_to_tensor(other, self))

    def __pow__(self, power, modulo=None) -> Tensor:
        from .ops import pow, utils

        return pow(self, utils.convert_to_tensor(power, self))

    def __matmul__(self, other) -> Tensor:
        from .ops import matmul, utils

        return matmul(self, utils.convert_to_tensor(other, self))

    def __invert__(self) -> Tensor:
        from .ops import bitwise_invert

        return bitwise_invert(self)

    def __and__(self, other) -> Tensor:
        from .ops import bitwise_and, utils

        return bitwise_and(self, utils.convert_to_tensor(other, self))

    def __or__(self, other):
        from .ops import bitwise_or, utils

        return bitwise_or(self, utils.convert_to_tensor(other, self))

    def __xor__(self, other):
        from .ops import bitwise_xor, utils

        return bitwise_xor(self, utils.convert_to_tensor(other, self))

    def __lshift__(self, other):
        from .ops import bitwise_left_shift, utils

        return bitwise_left_shift(self, utils.convert_to_tensor(other, self))

    def __rshift__(self, other):
        from .ops import bitwise_right_shift, utils

        return bitwise_right_shift(self, utils.convert_to_tensor(other, self))

    def __lt__(self, other):
        from .ops import less, utils

        return less(self, utils.convert_to_tensor(other, self))

    def __le__(self, other):
        from .ops import less_equal, utils

        return less_equal(self, utils.convert_to_tensor(other, self))

    def __gt__(self, other):
        from .ops import greater, utils

        return greater(self, utils.convert_to_tensor(other, self))

    def __eq__(self, other):
        from .ops import equal, utils

        return equal(self, utils.convert_to_tensor(other, self))

    def __ne__(self, other):
        from .ops import not_equal, utils

        return not_equal(self, utils.convert_to_tensor(other, self))

    def __radd__(self, other):
        from .ops import add

        return add(other, self)

    def __rsub__(self, other):
        from .ops import subtract

        return subtract(other, self)

    def __rmul__(self, other):
        from .ops import multiply

        return multiply(other, self)

    def __abs__(self):
        from .ops import abs

        return abs(self)

    def __bool__(self) -> bool:
        if self.size > 1:
            raise RuntimeError('Boolean value of Tensor with more than one value is ambiguous')
        return bool(self.item())

    def __float__(self) -> float:
        if self.size > 1:
            raise RuntimeError('only one element tensors can be converted to Python scalars')
        return float(self.item())

    def __index__(self) -> int:
        if self.size > 1:
            raise RuntimeError('only one element tensors can be converted to Python scalars')
        return int(self.item())

    def __int__(self) -> int:
        if self.size > 1:
            raise RuntimeError('only one element tensors can be converted to Python scalars')
        return int(self.item())

    def __str__(self):
        head = self.signature()
        if self.storage:
            array_str = str(self.cpu().numpy())
            return '{}\n{}'.format(head, array_str)
        else:
            if self.trace is None:
                return head
            else:
                return '{}\nfrom {}'.format(head, self.trace)

    def __getitem__(self, item):
        from hidet.graph.ops import strided_slice

        if isinstance(item, list):
            item = tuple(item)

        if not isinstance(item, tuple):
            item = tuple([item])

        # now, the item could have
        # 1. integer index
        # 2. slice
        # 3. Ellipsis
        # 4. None
        # e.g., [1, 3:5, ..., None]

        # process Ellipsis
        # e.g., x[1, ..., 2] -> x[1, :, :, 2]
        if Ellipsis in item:
            if item.count(Ellipsis) > 1:
                raise ValueError('Only one ellipsis allowed in index.')
            ellipsis_index = item.index(Ellipsis)
            ellipsis_ndim = len(self.shape) - sum([1 if axis not in [None, Ellipsis] else 0 for axis in item])
            ellipsis_ndim = max(ellipsis_ndim, 0)
            item = item[:ellipsis_index] + (slice(None),) * ellipsis_ndim + item[ellipsis_index + 1 :]

        # process None
        # e.g., x[2, None, 3] -> x[2, 1, 3]
        if None in item:
            dims = []
            for i, v in enumerate(item):
                if v is None:
                    dims.append(i)
            item = [v if v is not None else slice(None) for v in item]
            return self.unsqueeze(dims)[item]

        assert None not in item

        # normalize index
        normalized_item = []
        for i, v in enumerate(item):
            if isinstance(v, int):
                if v < 0:
                    v += self.shape[i]
                if v < 0 or v >= self.shape[i]:
                    raise IndexError(
                        'index {} is out of bound for dimension {} with size {}'.format(v, i, self.shape[i])
                    )
                normalized_item.append(v)
            else:
                normalized_item.append(v)
        item = tuple(normalized_item)

        # process slice and integer index
        rank = len(self.shape)
        while len(item) < rank:
            item = item + (slice(None),)
        starts, ends, steps = [], [], []
        squeeze_dims = []
        for dim, v in enumerate(item):
            if isinstance(v, int):
                squeeze_dims.append(dim)
                starts.append(v)
                ends.append(v + 1)
                steps.append(1)
            else:
                assert isinstance(v, slice)
                starts.append(v.start)
                ends.append(v.stop)
                steps.append(v.step)
        sliced = strided_slice(self, starts, ends, strides=steps).squeeze(squeeze_dims)
        return sliced

    def __iter__(self):
        raise TypeError('hidet.Tensor does not support iteration.')

    def __hash__(self):
        """
        Notes
        -----
        This is a hack to make hidet.Tensor hashable. There are some places where we need to use tensor as the key
        of a dictionary, (e.g., in a graph optimization pass or graph execution). However, to implement a correct
        protocol for hashable objects, we need to implement __eq__ as well to compare two objects when their hash
        values are equal. But as a tensor, the __eq__ method is used to do element-wise comparison, and it returns
        a tensor instead of a boolean value. Thus, there will be a problem when the dict that takes Tensor as key type
        has other kinds of objects (e.g., int). We deliberately ignore this problem here in exchange for the convenience
        of using tensor as the key of a dict.
        """
        return id(self)

    def __getstate__(self):
        if self.storage:
            data = self.detach().numpy()
        else:
            data = None

        return {
            'shape': self.shape,
            'dtype': self.dtype,
            'device': self.device,
            'data': data,
            'layout': self.layout,
            'trace': self.trace,
        }

    def __setstate__(self, state):
        data = state['data']
        if data is not None:
            assert isinstance(data, np.ndarray)
            tensor = from_numpy(data).to(device=state['device'])
            storage = tensor.storage
        else:
            storage = None

        self._device = state['device']
        self._shape = state['shape']
        self._dtype = state['dtype']
        self._storage = storage
        self._layout = state['layout']
        self._trace = state['trace']

    def __dlpack__(self, stream: Optional[int] = None):
        """
        This function is used to support interoperability with other frameworks that support __dlpack__ protocol.
        """
        from .impl.dlpack import to_dlpack

        if stream is not None:
            consumer_stream = hidet.cuda.ExternalStream(stream)
            provider_stream = hidet.cuda.current_stream()
            if consumer_stream != provider_stream:
                event = hidet.cuda.Event()
                event.record(provider_stream)
                consumer_stream.wait_event(event)
        return to_dlpack(self)

    def __dlpack_device__(self) -> Tuple[int, int]:
        """
        This function is used to support interoperability with other frameworks that support __dlpack__ protocol.
        """
        from .impl.dlpack import to_dlpack_device

        return to_dlpack_device(self)

    def tolist(self):
        """
        Convert the tensor to a nested list of numbers.

        Returns
        -------
        ret: the nested list of numbers
            The nested list of numbers. The number of nested levels is equal to the rank of the tensor.
        """
        return self.cpu().numpy().tolist()

    def to_device(self, device, /, *, stream=None):
        """
        Move the tensor to the specified device.

        Parameters
        ----------
        device: Device or str
            The device to move the tensor to.

        stream: Stream or None
            The stream to use for the copy. If None, the current stream is used.

        Returns
        -------
        ret: Tensor
            The tensor on the specified device.
        """
        device = instantiate_device(device)
        if device.is_cpu():
            tensor = self.cpu()
        elif device.is_cuda():
            tensor = self.cuda_async(device, stream=stream)
        else:
            raise ValueError('Cannot recognize device {}'.format(device))
        return tensor

    def item(self) -> Union[int, float, bool]:
        """
        Convert the tensor to a scalar value.

        Returns
        -------

        """
        if prod(self._shape) == 1:
            ret = self.squeeze(dims=list(range(len(self.shape)))).tolist()
            if not isinstance(ret, (int, float, bool)):
                raise TypeError('Cannot convert tensor to scalar.')
            return ret
        else:
            raise RuntimeError('Only support .item() method for tensor with only one element')

    def signature(self) -> str:
        """Get the signature of the tensor.

        Returns
        -------
        ret: str
            The signature of the tensor.
        """
        return "Tensor(shape={}, dtype='{}', device='{}')".format(self.shape, self.dtype.name, self.device)

    def is_symbolic(self) -> bool:
        """
        Check if the tensor is symbolic.

        A tensor is symbolic if it is not backed by any storage (i.e., ``self.storage is None``).

        Returns
        -------
        ret: bool
            True if the tensor is symbolic, False otherwise.
        """
        return self.storage is None

    def contiguous(self):
        """Create a tensor with contiguous row-major layout.

        If the tensor already has the continuous row-major layout, this tensor is returned directly.

        Returns
        -------
        ret: Tensor
            The tensor with contiguous row-major layout.
        """
        if isinstance(self.layout, RowMajorLayout):
            return self
        return self.reshape(self.shape)

    def reshape(self, shape: Sequence[int]):
        """Create a reshaped tensor.

        See Also :func:`hidet.graph.ops.reshape`.

        Parameters
        ----------
        shape: Sequence[int]
            The new shape.

        Returns
        -------
        ret: Tensor
            The reshaped tensor.
        """
        from .ops import reshape

        return reshape(self, shape)

    def squeeze(self, dims: Union[int, Sequence[int]]):
        """Create a squeezed tensor.

        See Also :func:`hidet.graph.ops.squeeze`.

        Parameters
        ----------
        dims: Union[int, Sequence[int]]
            The dimension(s) to squeeze.

        Returns
        -------
        ret: Tensor
            The squeezed tensor.
        """
        from .ops import squeeze

        return squeeze(self, dims)

    def unsqueeze(self, dims: Union[int, Sequence[int]]):
        """Create a unsqueezed tensor.

        See Also :func:`hidet.graph.ops.unsqueeze`.

        Parameters
        ----------
        dims: Union[int, Sequence[int]]
            The dimensions to unsqueeze.

        Returns
        -------
        ret: Tensor
            The unsqueezed tensor.
        """
        from .ops import unsqueeze

        return unsqueeze(self, dims)

    def rearrange(self, plan: List[List[int]]):
        """Create a rearranged tensor.

        See Also :func:`hidet.graph.ops.rearrange`.

        Parameters
        ----------
        plan: List[List[int]]
            The rearrange plan.

        Returns
        -------
        ret: Tensor
            The rearranged tensor.
        """
        from .ops import rearrange

        return rearrange(self, plan)

    def sum(self, dims: Union[int, List[int]], keep_dim: bool = False):
        """Create a sum reduced tensor.

        See Also :func:`hidet.graph.ops.reduce_sum`.

        Parameters
        ----------
        dims: Union[int, List[int]]
            The dimensions to sum up.

        keep_dim: bool
            Whether to keep the reduced dimensions.

        Returns
        -------
        ret: Tensor
            The reduced tensor.
        """
        from .ops import sum

        return sum(self, dims=dims, keep_dim=keep_dim)

    def mean(self, dims: Union[int, List[int]], keep_dim: bool = False):
        """Create a mean reduced tensor.

        See Also :func:`hidet.graph.ops.reduce_mean`.

        Parameters
        ----------
        dims: Union[int, List[int]]
            The dimensions to average up.

        keep_dim: bool
            Whether to keep the reduced dimensions.

        Returns
        -------
        ret: Tensor
            The reduced tensor.
        """
        from .ops import mean

        return mean(self, dims=dims, keep_dim=keep_dim)

    def astype(self, dtype):
        """Cast the data type of current tensor.

        Parameters
        ----------
        dtype: DataType or str
            The target data type to convert to.

        Returns
        -------
        ret: Tensor
            The tensor with the new data type.
        """
        from .ops import cast

        return cast(self, dtype)

    def to(self, dtype=None, device=None):
        """Cast the data type of current tensor or/and move it to another device.

        Parameters
        ----------
        dtype: DataType or str, optional
            The target data type to convert to. None indicates unchanged.

        device: Device or str, optional
            The target device to copy the tensor. None indicates unchanged.

        Returns
        -------
        ret: Tensor
            The tensor with the new data type on target device.
        """
        from .ops import cast

        tensor = self
        if dtype is not None:
            tensor = cast(tensor, dtype)
        if device is not None:
            device = instantiate_device(device)
            if device.is_cpu():
                tensor = tensor.cpu()
            elif device.is_cuda():
                tensor = tensor.cuda(device)
            else:
                raise ValueError('Cannot recognize device {}'.format(device))
        return tensor

    def cpu(self):
        """Create a copy of self tensor on cpu device.

        If the current tensor is already on cpu device, self is returned.

        Returns
        -------
        ret: Tensor
            The new tensor or self.
        """
        if self.device.type == 'cpu':
            return self
        else:
            if self.trace is None:
                return Tensor(self.shape, self.dtype, 'cpu', self.storage.cpu() if self.storage else None, self.layout)
            else:
                raise ValueError('Please use .detach() to detach a trace variable first.')

    def cuda(self, device=None):
        """Create a copy of self tensor on cuda device.

        If the current tensor is already on cuda device, self is returned.

        Parameters
        ----------
        device: Device, optional
            The target cuda device. None indicates the current cuda device.

        Returns
        -------
        ret: Tensor
            The new tensor or self.
        """
        if device is None:
            device = 'cuda'
        device = instantiate_device(device)
        if self.device == device:
            return self
        else:
            if self.trace is None:
                return Tensor(
                    self.shape, self.dtype, device, self.storage.cuda(device.id) if self.storage else None, self.layout
                )
            else:
                raise ValueError('Please use .detach() to detach a trace variable first.')

    def copy(self) -> Tensor:
        """Create a copy of current tensor.

        Returns
        -------
        ret: Tensor
            A new tensor with the same contents as the current one.
        """
        if self.trace is not None:
            raise ValueError('Please use .detach() to detach a trace variable first before copying.')
        return Tensor(
            shape=list(self.shape),
            dtype=self.dtype,
            device=self.device,
            storage=self.storage.copy(),
            layout=self.layout,
            trace=None,
        )

    def copy_async(self, stream=None) -> Tensor:
        """Create a copy of current tensor asynchronously.

        Parameters
        ----------
        stream: hidet.cuda.Stream, optional
            The stream to copy the tensor. None indicates the current stream of the device where self tensor is on.

        Returns
        -------
        ret: Tensor
            A new tensor with the same contents as the current one.
        """
        if self.trace is not None:
            raise ValueError('Please use .detach() to detach a trace variable first before copying.')
        return Tensor(
            shape=list(self.shape),
            dtype=self.dtype,
            device=self.device,
            storage=self.storage.copy_async(stream),
            layout=self.layout,
            trace=None,
        )

    def detach(self):
        """Detach the current tensor from tracing.

        Returns
        -------
        ret: Tensor
            The detached tensor.
        """
        if self.trace is None:
            return self
        else:
            return Tensor(
                shape=self.shape,
                dtype=self.dtype,
                device=self.device,
                storage=self.storage,
                layout=self.layout,
                trace=None,
            )

    def cpu_async(self, stream=None):
        """
        Copy the tensor to CPU asynchronously.

        Parameters
        ----------
        stream: hidet.cuda.Stream, optional
            The stream to copy the tensor to CPU on.

        Returns
        -------
        ret: Tensor
            The tensor on CPU.
        """
        if self.device.type == 'cpu':
            return self
        else:
            if self.trace is None:
                ret = Tensor(
                    self.shape, self.dtype, 'cpu', self.storage.cpu_async(stream) if self.storage else None, self.layout
                )
                return ret
            else:
                raise ValueError('Please use .detach() to detach a trace variable first.')

    def cuda_async(self, device=None, stream=None):
        """
        Copy the tensor to GPU asynchronously.

        Parameters
        ----------
        device: Device, optional
            The target cuda device. None indicates the current cuda device.

        stream: hidet.cuda.Stream, optional
            The stream to copy the tensor to GPU on. None indicates the current stream.

        Returns
        -------
        ret: Tensor
            The tensor on GPU.
        """
        if device is None:
            device = 'cuda'
        device = instantiate_device(device)
        if self.device.is_cuda():
            return self
        else:
            if self.trace is None:
                ret = Tensor(
                    self.shape,
                    self.dtype,
                    device,
                    self.storage.cuda_async(device.id, stream) if self.storage else None,
                    self.layout,
                )
                return ret
            else:
                raise ValueError('Please use .detach() to detach a trace variable first.')

    def numpy(self) -> np.ndarray:
        """
        Convert the tensor to a numpy array.

        The tensor must be on CPU device. Otherwise, a RuntimeError will be raised. The returned numpy array will share
        the same memory with the tensor.

        Returns
        -------
        ret: np.ndarray
            The numpy array.
        """
        if self.device.type != 'cpu':
            raise RuntimeError('Cannot convert a tensor on {} to numpy array.'.format(self.device))
        if self.dtype in [dtypes.bfloat16, dtypes.tfloat32]:
            warnings.warn('numpy does not support {}, converting to float32'.format(self.dtype.name))
            return self.astype(dtypes.float32).numpy()
        if self.dtype == dtypes.boolean:
            # workaround for numpy not supporting exporting boolean to dlpack
            return np.from_dlpack(self.to(dtype='uint8')).astype(np.bool_)
        else:
            return np.from_dlpack(self)

    def torch(self):
        """
        Convert to a torch tensor.

        Returns
        -------
        ret: torch.Tensor
            The torch tensor that shares the memory with the hidet tensor.
        """
        import torch

        return torch.from_dlpack(self)


def empty(shape, dtype='float32', device='cpu', layout=None):
    """Create an uninitialized tensor.

    Parameters
    ----------
    shape: Sequence[int]
        The shape of new tensor.

    dtype: str or DataType
        The data type of element of the tensor.

    device: Device or str, default 'cpu'
        The device of the new tensor is created on.

    layout: DataLayout, optional
        The layout of the new tensor. None indicates the default layout (row-major layout).

    Returns
    -------
    ret: Tensor
        The created tensor.
    """
    dtype = data_type(dtype)
    num_bytes = prod(shape) * dtype.nbytes
    storage = Storage.new(device, num_bytes)
    return Tensor(shape=shape, dtype=dtype, device=device, storage=storage, layout=layout)


def symbol(shape: Sequence[int], dtype='float32', device='cpu', layout=None) -> Tensor:
    """Create a symbolic tensor.

    Parameters
    ----------
    shape: Sequence[int]
        The shape of new tensor.

    dtype: str
        The data type of element of the tensor.

    device: Device or str, default 'cpu'
        The device of the new tensor is created on.

    layout: DataLayout, optional
        The layout of the new tensor. None indicates the default layout (row-major layout).

    Returns
    -------
    ret: Tensor
        The created tensor.

    """
    return Tensor(shape=shape, dtype=dtype, device=device, storage=None, layout=layout)


def zeros(shape: Sequence[int], dtype='float32', device='cpu') -> Tensor:
    """Create a tensor initialized with zero.

    Parameters
    ----------
    shape: Sequence[int]
        The shape of new tensor.

    dtype: str
        The data type of element of the tensor.

    device: Device or str, default 'cpu'
        The device of the new tensor is created on.

    Returns
    -------
    ret: Tensor
        The created tensor.
    """
    dtype = data_type(dtype)
    return full(shape, dtype.zero, dtype, device)


def ones(shape, dtype='float32', device='cpu') -> Tensor:
    """Create a tensor initialized with one.

    Parameters
    ----------
    shape: Sequence[int]
        The shape of new tensor.

    dtype: DataType or str, default 'float32'
        The data type of element of the tensor.

    device: Device or str, default 'cpu'
        The device of the new tensor is created on.

    Returns
    -------
    ret: Tensor
        The created tensor.
    """
    dtype = data_type(dtype)
    return full(shape, dtype.one, dtype, device)


def full(shape, fill_value: Union[float, int], dtype='float32', device='cpu') -> Tensor:
    """Create a tensor initialized with given constant.

    Parameters
    ----------
    shape: Sequence[int]
        The shape of new tensor.

    fill_value: float or int or hidet.ir.Constant
        The constant to initialize the new tensor.

    dtype: DataType or str, default 'float32'
        The data type of element of the tensor.

    device: Device or str, default 'cpu'
        The device of the new tensor is created on.

    Returns
    -------
    ret: Tensor
        The created tensor.
    """
    from hidet import ops

    dtype = data_type(dtype)
    return ops.full(shape=shape, value=fill_value, dtype=dtype, device=device)


def randn(shape, dtype='float32', mean=0.0, stddev=1.0, device='cpu') -> Tensor:
    """Create a tensor with uniformly distributed values.

    Parameters
    ----------
    shape: Sequence[int]
        The shape of new tensor.

    dtype: DataType or str, default 'float32'
        The data type of element of the tensor.

    mean: float, default 0.0
        The mean of the uniform distribution.

    stddev: float, default 1.0
        The standard deviation of the uniform distribution.

    device: Device or str, default 'cpu'
        The device of the new tensor is created on.

    Returns
    -------
    ret: Tensor
        The created tensor.

    Examples
    --------
    >>> randn([2, 3])
    Tensor(shape=[2, 3], dtype='float32', device='cuda')
    [[ 0.10720467 -1.6906018   0.06347568]
     [-0.37061226  0.562728    1.857547  ]]
    """

    np_tensor = np.random.randn(*shape).astype(np.float32)
    np_tensor = np_tensor * stddev + mean
    hidet_tensor = from_numpy(np_tensor)
    return hidet_tensor.to(device=device, dtype=dtype)


def randint(low: int, high=None, shape: Sequence[int] = (), dtype: str = 'int32') -> Tensor:
    dtype_map = {'int32': np.int32, 'int64': np.int64}
    if dtype not in dtype_map:
        return randint(low=low, high=high, shape=shape, dtype='int32').astype(dtype)
    return asarray(np.random.randint(low=low, high=high, size=shape, dtype=dtype_map[dtype]))


def empty_like(data, shape=None, dtype=None, device=None, layout=None) -> Tensor:
    """
    Create an uninitialized tensor with the same shape, dtype, and device as the given tensor.

    Parameters
    ----------
    data: Tensor
        The tensor to copy shape, dtype, and device from.

    shape: Sequence[int], optional
        The shape of new tensor. If None, the shape of data is used.

    dtype: DataType or str, optional
        The data type of element of the tensor. If None, the dtype of data is used.

    device: Device or str, optional
        The device of the new tensor is created on. If None, the device of data is used.

    layout: DataLayout, optional
        The layout of the new tensor. If None, the layout of data is used.

    Returns
    -------
    ret: Tensor
        The created tensor.
    """
    return empty(
        shape=shape if shape is not None else data.shape,
        dtype=dtype if dtype is not None else data.dtype,
        device=device if device is not None else data.device,
        layout=layout if layout is not None else data.layout,
    )


def symbol_like(data, shape=None, dtype=None, device=None, layout=None):
    """Create a symbol tensor like an existing tensor.

    Parameters
    ----------
    data: Tensor
        The tensor to copy shape, dtype, and device from.

    shape: Sequence[int], optional
        The shape of new tensor. If None, the shape of data is used.

    dtype: DataType or str, optional
        The data type of element of the tensor. If None, the dtype of data is used.

    device: Device or str, optional
        The device of the new tensor is created on. If None, the device of data is used.

    layout: DataLayout, optional
        The layout of the new tensor. If None, the layout of data is used.

    Returns
    -------
    ret: Tensor
        The created symbol tensor.
    """
    return symbol(
        shape=shape if shape is not None else data.shape,
        dtype=dtype if dtype is not None else data.dtype,
        device=device if device is not None else data.device,
        layout=layout if layout is not None else data.layout,
    )


def zeros_like(data, shape=None, dtype=None, device=None) -> Tensor:
    """
    Create a tensor initialized with zero with the same shape, dtype, and device as the given tensor.

    Parameters
    ----------
    data: Tensor
        The tensor to copy shape, dtype, and device from.

    shape: Sequence[int], optional
        The shape of new tensor. If None, the shape of data is used.

    dtype: DataType or str, optional
        The data type of element of the tensor. If None, the dtype of data is used.

    device: Device or str, optional
        The device of the new tensor is created on. If None, the device of data is used.

    Returns
    -------
    ret: Tensor
        The created tensor with all elements as zero.
    """
    return zeros(
        shape=data.shape if shape is None else shape,
        dtype=data.dtype if dtype is None else dtype,
        device=data.device if device is None else device,
    )


def ones_like(data, shape=None, dtype=None, device=None) -> Tensor:
    """
    Create a tensor initialized with one with the same shape, dtype, and device as the given tensor.

    Parameters
    ----------
    data: Tensor
        The tensor to copy shape, dtype, and device from.

    shape: Sequence[int], optional
        The shape of new tensor. If None, the shape of data is used.

    dtype: DataType or str, optional
        The data type of element of the tensor. If None, the dtype of data is used.

    device: Device or str, optional
        The device of the new tensor is created on. If None, the device of data is used.

    Returns
    -------
    ret: Tensor
        The created tensor with all elements as one.
    """
    return ones(
        shape=data.shape if shape is None else shape,
        dtype=data.dtype if dtype is None else dtype,
        device=data.device if device is None else device,
    )


def full_like(
    data: Tensor,
    fill_value,
    shape: Optional[Sequence[int]] = None,
    dtype: Optional[str] = None,
    device: Optional[str] = None,
) -> Tensor:
    """
    Create a tensor initialized with fill_value with the same shape, dtype, and device as the given tensor.

    Parameters
    ----------
    data: Tensor
        The tensor to copy shape, dtype, and device from.

    fill_value: int, float, or bool
        The value to fill the tensor with.

    shape: Sequence[int], optional
        The shape of new tensor. If None, the shape of data is used.

    dtype: DataType or str, optional
        The data type of element of the tensor. If None, the dtype of data is used.

    device: Device or str, optional
        The device of the new tensor is created on. If None, the device of data is used.

    Returns
    -------
    ret: Tensor
        The created tensor with all elements as fill_value.
    """
    return full(
        shape=data.shape if shape is None else shape,
        fill_value=fill_value,
        dtype=data.dtype if dtype is None else dtype,
        device=data.device if device is None else device,
    )


def randn_like(
    data: Tensor, shape: Optional[Sequence[int]] = None, dtype: Optional[str] = None, device: Optional[str] = None
) -> Tensor:
    """
    Create a randomly initialized tensor with the same shape, dtype, and device as the given tensor.

    Parameters
    ----------
    data: Tensor
        The tensor to copy shape, dtype, and device from.

    shape: Sequence[int], optional
        The shape of new tensor. If None, the shape of data is used.

    dtype: DataType or str, optional
        The data type of element of the tensor. If None, the dtype of data is used.

    device: Device or str, optional
        The device of the new tensor is created on. If None, the device of data is used.

    Returns
    -------
    ret: Tensor
        The created tensor with random values sampled from a normal distribution.
    """
    return randn(
        shape=data.shape if shape is None else shape,
        dtype=data.dtype if dtype is None else dtype,
        device=data.device if device is None else device,
    )


def from_numpy(nparray: np.ndarray) -> Tensor:
    """
    Create a tensor from a numpy array, sharing the memory with the numpy array when possible.

    Parameters
    ----------
    nparray: numpy.ndarray
        The numpy array to create the tensor from.

    Returns
    -------
    ret: Tensor
        The created tensor.
    """
    if not isinstance(nparray, np.ndarray):
        raise TypeError('nparray must be a numpy array')
    if not nparray.flags['WRITEABLE']:
        # make a copy if the array is read-only
        nparray = nparray.copy()
    if nparray.dtype == np.bool_:
        return from_dlpack(nparray.astype(np.uint8)).to(dtype='bool')
    else:
        return from_dlpack(nparray)


def from_dlpack(dltensor) -> Tensor:
    """
    Create a hidet tensor from an object that implements the __dlpack__ protocol.

    Parameters
    ----------
    dltensor: an object that implements the DLPack protocol.
        The object must have the method `__dlpack__` that returns a PyCapsule object with name `dltensor`.

    Returns
    -------
    ret: Tensor
        The hidet tensor that shares the same storage with the DLPack tensor.
    """
    from .impl.dlpack import from_dlpack_capsule

    if not hasattr(dltensor, '__dlpack__'):
        raise RuntimeError('Expect a dltensor that implements __dlpack__ method.')
    return from_dlpack_capsule(dltensor.__dlpack__())


def from_torch(torch_tensor):
    """Create a hidet tensor from pytorch tensor.

    The created tensor shared the same memory as given pytorch tensor. Thus, any content
    modification on one tensor would be reflected on the other one.

    Parameters
    ----------
    torch_tensor: torch.Tensor
        The pytorch tensor.

    Returns
    -------
    ret: Tensor
        The created hidet tensor.
    """
    import torch

    if not isinstance(torch_tensor, torch.Tensor):
        raise ValueError('Expect a torch.Tensor, got {}'.format(type(torch_tensor)))
    if torch_tensor.requires_grad:
        raise ValueError('Please first call .detach() on the pytorch tensor before converting it to hidet.')
    assert isinstance(torch_tensor, torch.Tensor)
    if torch_tensor.dtype == torch.bool:
        # exporting torch.bool to dlpack is not supported by pytorch yet
        return from_dlpack(torch_tensor.to(dtype=torch.uint8)).to(dtype='bool')
    else:
        return from_dlpack(torch_tensor)


def asarray(obj, /, *, dtype=None, device=None) -> Tensor:
    """
    Convert a list, tuple, or numpy ndarray to a hidet tensor.

    Parameters
    ----------
    obj: Union[bool, int, float, List, Tuple, Tensor, np.ndarray]
        The object to be converted.

    dtype: DataType, optional
        The data type of the output tensor.

    device: Device or str
        The device of the output tensor.

    Returns
    -------
    ret: Tensor
        The hidet tensor converted from given object.
    """
    from hidet.ir.dtypes import dtype_to_numpy

    if isinstance(obj, Tensor):
        ret = obj
    elif isinstance(obj, np.ndarray):
        ret = from_numpy(obj)
    else:
        array = np.array(obj, dtype=dtype_to_numpy(dtype) if dtype else None)
        if array.dtype == np.float64:
            # numpy uses float64 as the default float data type, convert it to float32 as hidet takes float32 as default
            array = array.astype(np.float32)
        ret = from_numpy(array)
    return ret.to(dtype=dtype, device=device)
