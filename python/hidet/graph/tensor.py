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
from hidet.runtime.device import Device, instantiate_device


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

        self.shape: List[int] = [int(v) for v in shape]
        self.dtype: DataType = data_type(dtype)
        self.device: Device = instantiate_device(device)
        self.storage: Optional[Storage] = storage
        self.layout: DataLayout = layout if layout else DataLayout.row_major(shape)
        self.trace: Optional[Tuple[Operator, int]] = trace

    def __pos__(self):
        return self

    def __neg__(self) -> Tensor:
        from .ops import neg

        return neg(self)

    def __add__(self, other) -> Tensor:
        from .ops import add

        return add(self, other)

    def __sub__(self, other) -> Tensor:
        from .ops import sub

        return sub(self, other)

    def __mul__(self, other) -> Tensor:
        from .ops import multiply

        return multiply(self, other)

    def __truediv__(self, other) -> Tensor:
        from .ops import divide

        return divide(self, other)

    def __floordiv__(self, other) -> Tensor:
        raise NotImplementedError()

    def __mod__(self, other) -> Tensor:
        raise NotImplementedError()

    def __pow__(self, power, modulo=None) -> Tensor:
        raise NotImplementedError()

    def __matmul__(self, other) -> Tensor:
        from .ops import matmul

        return matmul(self, other)

    def __invert__(self) -> Tensor:
        from .ops import bitwise_not

        return bitwise_not(self)

    def __and__(self, other) -> Tensor:
        from .ops import bitwise_and

        return bitwise_and(self, other)

    def __or__(self, other):
        from .ops import bitwise_or

        return bitwise_or(self, other)

    def __xor__(self, other):
        from .ops import bitwise_xor

        return bitwise_xor(self, other)

    def __lshift__(self, other):
        from .ops import leftshift

        return leftshift(self, other)

    def __rshift__(self, other):
        from .ops import rightshift

        return rightshift(self, other)

    def __lt__(self, other):
        from .ops import less_than

        return less_than(self, other)

    def __le__(self, other):
        from .ops import less_or_equal

        return less_or_equal(self, other)

    def __gt__(self, other):
        from .ops import greater_than

        return greater_than(self, other)

    def __eq__(self, other):
        from .ops import equal

        return equal(self, other)

    def __ne__(self, other):
        raise NotImplementedError()

    def __iadd__(self, other):
        raise NotImplementedError()

    def __isub__(self, other):
        raise NotImplementedError()

    def __imul__(self, other):
        raise NotImplementedError()

    def __itruediv__(self, other):
        raise NotImplementedError()

    def __ifloordiv__(self, other):
        raise NotImplementedError()

    def __imod__(self, other):
        raise NotImplementedError()

    def __ipow__(self, other):
        raise NotImplementedError()

    def __imatmul__(self, other):
        raise NotImplementedError()

    def __iand__(self, other):
        raise NotImplementedError()

    def __ior__(self, other):
        raise NotImplementedError()

    def __ixor__(self, other):
        raise NotImplementedError()

    def __ilshift__(self, other):
        raise NotImplementedError()

    def __irshift__(self, other):
        raise NotImplementedError()

    def __radd__(self, other):
        from .ops import add

        return add(other, self)

    def __rsub__(self, other):
        from .ops import sub

        return sub(other, self)

    def __rmul__(self, other):
        from .ops import multiply

        return multiply(other, self)

    def __rtruediv__(self, other):
        raise NotImplementedError()

    def __rfloordiv__(self, other):
        raise NotImplementedError()

    def __rmod__(self, other):
        raise NotImplementedError()

    def __rpow__(self, other):
        raise NotImplementedError()

    def __rmatmul__(self, other):
        raise NotImplementedError()

    def __rand__(self, other):
        raise NotImplementedError()

    def __ror__(self, other):
        raise NotImplementedError()

    def __rxor__(self, other):
        raise NotImplementedError()

    def __rlshift__(self, other):
        raise NotImplementedError()

    def __rrshift__(self, other):
        raise NotImplementedError()

    def __abs__(self):
        from .ops import abs

        return abs(self)

    def __bool__(self) -> bool:
        raise NotImplementedError()

    def __array_namespace__(self, *, api_version=None):
        raise NotImplementedError()

    def __complex__(self) -> complex:
        raise NotImplementedError()

    def __float__(self) -> float:
        raise NotImplementedError()

    def __index__(self) -> int:
        raise NotImplementedError()

    def __int__(self) -> int:
        raise NotImplementedError()

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

        # process slice and integer index
        rank = len(self.shape)
        if all(not isinstance(v, slice) for v in item) and len(item) == rank:
            # element access, return a python scalar
            return strided_slice(self, starts=list(item), ends=[v + 1 for v in item]).numpy().flatten()[0]
        else:
            # slice access, return a hidet tensor
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
                    starts.append(v.start if v.start is not None else 0)
                    ends.append(v.stop if v.stop is not None else self.shape[dim])
                    steps.append(v.step if v.step is not None else 1)
            sliced = strided_slice(self, starts, ends, strides=steps).squeeze(squeeze_dims)
            return sliced

    def __setitem__(self, key, value):
        raise NotImplementedError()

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
            tensor = from_numpy(data)
            if state['device'] == 'cuda':
                tensor = tensor.cuda()
            storage = tensor.storage
        else:
            storage = None

        self.shape = state['shape']
        self.dtype = state['dtype']
        self.device = state['device']
        self.storage = storage
        self.layout = state['layout']
        self.trace = state['trace']

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

    def to_device(self, device, /, *, stream=None):
        raise NotImplementedError()

    def signature(self) -> str:
        """Get the signature of the tensor.

        Returns
        -------
        ret: str
            The signature of the tensor.
        """
        return "Tensor(shape={}, dtype='{}', device='{}')".format(self.shape, self.dtype.name, self.device)

    def is_symbolic(self) -> bool:
        return self.storage is None

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
    def num_elements(self):
        """The number of elements of the tensor.

        Returns
        -------
        ret: int
            The number of elements.
        """
        return prod(self.shape)

    @property
    def op(self):
        """The operator that produces this tensor.

        Returns
        -------
        ret: hidet.graph.operator.Operator, optional
            The operator that produces this tensor. None indicates it is not traced.
        """
        return self.trace[0] if self.trace else None

    def scalar(self) -> Union[float, int]:
        """Get the scalar value.

        If a tensor has shape ``[]`` (i.e., rank = 0), this tensor is a scalar. This function get the value of this
        tensor.

        Returns
        -------
        ret: Union[float, int]
            The value of the tensor.
        """
        if len(self.shape) != 0:
            raise ValueError('Can not convert a Tensor with shape {} to a scalar.'.format(self.shape))
        value = self.cpu().numpy().tolist()
        assert isinstance(value, (int, float))
        return value

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

    def flatten(self, start_dim=0, end_dim=-1):
        """Create a flattened tensor.

        See Also :func:`hidet.graph.ops.flatten`.

        Parameters
        ----------
        start_dim: int
            The start dimension to flatten.

        end_dim: int
            The end dimension (inclusive) to flatten.

        Returns
        -------
        ret: Tensor
            The flattened tensor.
        """
        from .ops import flatten

        return flatten(self, start_dim, end_dim)

    def transpose(self, axes: Optional[Sequence[int]]):
        """Create a transposed tensor.

        See Also :func:`hidet.graph.ops.transpose`.

        Parameters
        ----------
        axes: Optional[Sequence[int]]
            The axes to transpose.

        Returns
        -------
        ret: Tensor
            The transposed tensor.
        """
        from .ops import transpose

        return transpose(self, axes)

    def barrier(self) -> Tensor:
        """Create a fusion barrier toward current tensor.

        See Also :func:`hidet.graph.ops.barrier`.

        Returns
        -------
        ret: Tensor
            The same tensor after barrier.
        """
        from .ops import barrier

        return barrier(self)

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
        from .ops import reduce_sum

        return reduce_sum(self, dims=dims, keep_dim=keep_dim)

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
        from .ops import reduce_mean

        return reduce_mean(self, dims=dims, keep_dim=keep_dim)

    def rsqrt(self):
        """Compute the ``1/sqrt(x)`` of current tensor x.

        See Also :func:`hidet.graph.ops.rsqrt`.

        Returns
        -------
        ret: Tensor
            The result tensor.
        """
        from .ops import rsqrt

        return rsqrt(self)

    def cast(self, dtype):
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

    def to(self, dtype: Optional[str] = None, device: Optional[Union[Device, str]] = None):
        """Cast the data type of current tensor or move it to another device.

        Parameters
        ----------
        dtype: str, optional
            The target data type to convert to. None indicates unchanged.

        device: Union[Device, str], optional
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

    def copy_async(self, stream: Optional[hidet.cuda.Stream] = None) -> Tensor:
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
        if self.device == 'cuda':
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
            return self.cast(dtypes.float32).numpy()
        if self.dtype == dtypes.boolean:
            # workaround for numpy not supporting exporting boolean to dlpack
            return np.from_dlpack(self.to(dtype='uint8')).astype(np.bool)
        else:
            return np.from_dlpack(self)
        # storage = self.contiguous().storage  # convert if this tensor is not in row major layout
        # np_array = storage.as_array(num_elements=prod(self.shape), dtype=self.dtype, share_mem=share_mem)
        # np_array: np.ndarray = np_array.reshape(self.shape)
        # return np_array

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


def empty(shape, dtype='float32', device='cuda', layout=None):
    """Create an uninitialized tensor.

    Parameters
    ----------
    shape: Sequence[int]
        The shape of new tensor.

    dtype: str or DataType
        The data type of element of the tensor.

    device: Device or str
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


def symbol(shape: Sequence[int], dtype='float32', device='cuda', layout=None) -> Tensor:
    """Create a symbolic tensor.

    Parameters
    ----------
    shape: Sequence[int]
        The shape of new tensor.

    dtype: str
        The data type of element of the tensor.

    device: str
        The device of the new tensor is created on.

    layout: DataLayout, optional
        The layout of the new tensor. None indicates the default layout (row-major layout).

    Returns
    -------
    ret: Tensor
        The created tensor.

    """
    return Tensor(shape=shape, dtype=dtype, device=device, storage=None, layout=layout)


def zeros(shape: Sequence[int], dtype='float32', device='cuda') -> Tensor:
    """Create a tensor initialized with zero.

    Parameters
    ----------
    shape: Sequence[int]
        The shape of new tensor.

    dtype: str
        The data type of element of the tensor.

    device: str
        The device of the new tensor is created on.

    Returns
    -------
    ret: Tensor
        The created tensor.
    """
    tensor = empty(shape, dtype, device)
    hidet.cuda.memset_async(addr=tensor.storage.addr, num_bytes=tensor.nbytes, value=0)
    return tensor


def ones(shape, dtype='float32', device='cuda') -> Tensor:
    """Create a tensor initialized with one.

    Parameters
    ----------
    shape: Sequence[int]
        The shape of new tensor.

    dtype: str
        The data type of element of the tensor.

    device: str
        The device of the new tensor is created on.

    Returns
    -------
    ret: Tensor
        The created tensor.
    """
    dtype = data_type(dtype)
    return full(shape, dtype.one, dtype, device)


def full(shape, fill_value: Union[float, int], dtype='float32', device='cuda') -> Tensor:
    """Create a tensor initialized with given constant.

    Parameters
    ----------
    shape: Sequence[int]
        The shape of new tensor.

    fill_value: float or int or hidet.ir.Constant
        The constant to initialize the new tensor.

    dtype: DataType or str
        The data type of element of the tensor.

    device: Device or str
        The device of the new tensor is created on.

    Returns
    -------
    ret: Tensor
        The created tensor.
    """
    from hidet import ops

    dtype = data_type(dtype)
    return ops.full(shape=shape, value=fill_value, dtype=dtype, device=device)


def randn(shape, dtype='float32', mean=0.0, stddev=1.0, device='cuda') -> Tensor:
    """Create a tensor with uniformly distributed values.

    Parameters
    ----------
    shape: Sequence[int]
        The shape of new tensor.

    dtype: str
        The data type of element of the tensor.

    mean: float
        The mean of the uniform distribution.

    stddev: float
        The standard deviation of the uniform distribution.

    device: str
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

    np_tensor = np.random.randn(*shape)
    np_tensor = np_tensor * stddev + mean
    hidet_tensor = from_numpy(np_tensor)
    return hidet_tensor.to(device=device, dtype=dtype)

    # if device != 'cuda' or dtype != 'float32':
    #     return randn(shape, 'float32', mean, stddev, 'cuda', layout).to(device=device, dtype=dtype)
    #
    # assert device == 'cuda' and dtype == 'float32'
    # tensor = empty(shape, dtype='float32', device='cuda', layout=layout)
    # cuda.generate_normal(tensor.storage.addr, num_elements=prod(tensor.shape), mean=mean, stddev=stddev)
    # return tensor


def randint(low: int, high=None, shape: Sequence[int] = (), dtype: str = 'int32') -> Tensor:
    dtype_map = {'int32': np.int32, 'int64': np.int64}
    if dtype not in dtype_map:
        return randint(low=low, high=high, shape=shape, dtype='int32').cast(dtype)
    return asarray(np.random.randint(low=low, high=high, size=shape, dtype=dtype_map[dtype]))


def empty_like(
    data: Tensor,
    shape: Optional[Sequence[int]] = None,
    dtype: Optional[str] = None,
    device: Optional[str] = None,
    layout: Optional[DataLayout] = None,
) -> Tensor:
    return empty(
        shape=shape if shape is not None else data.shape,
        dtype=dtype if dtype is not None else data.dtype,
        device=device if device is not None else data.device,
        layout=layout if layout is not None else data.layout,
    )


def symbol_like(data: Tensor, shape=None, dtype=None, device=None, layout=None):
    """Create a symbol tensor like an existing tensor.

    Parameters
    ----------
    data: Tensor
        The information of this tensor will be used to create the symbol tensor.

    shape: Sequence[int], optional
        The shape of the new tensor.

    dtype: str, optional
        The data type of the new tensor.

    device: str, optional
        The device of the new tensor.

    layout: DataLayout, optional
        The data layout of the new tensor.

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


def zeros_like(
    data: Tensor, shape: Optional[Sequence[int]] = None, dtype: Optional[str] = None, device: Optional[str] = None
) -> Tensor:
    return zeros(
        shape=data.shape if shape is None else shape,
        dtype=data.dtype if dtype is None else dtype,
        device=data.device if device is None else device,
    )


def ones_like(
    data: Tensor, shape: Optional[Sequence[int]] = None, dtype: Optional[str] = None, device: Optional[str] = None
) -> Tensor:
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
    return full(
        shape=data.shape if shape is None else shape,
        fill_value=fill_value,
        dtype=data.dtype if dtype is None else dtype,
        device=data.device if device is None else device,
    )


def randn_like(
    data: Tensor, shape: Optional[Sequence[int]] = None, dtype: Optional[str] = None, device: Optional[str] = None
) -> Tensor:
    return randn(
        shape=data.shape if shape is None else shape,
        dtype=data.dtype if dtype is None else dtype,
        device=data.device if device is None else device,
    )


def from_numpy(nparray: np.ndarray) -> Tensor:
    if not isinstance(nparray, np.ndarray):
        raise TypeError('nparray must be a numpy array')
    if not nparray.flags['WRITEABLE']:
        # make a copy if the array is read-only
        nparray = nparray.copy()
    if nparray.dtype == np.bool:
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
    if isinstance(obj, Tensor):
        ret = obj
    else:
        ret = from_numpy(np.array(obj))
    return ret.to(dtype=dtype, device=device)


def arange(start, /, stop=None, step=1, *, dtype=None, device=None) -> Tensor:
    raise NotImplementedError()


def eye(n_rows, n_cols=None, /, *, k=0, dtype=None, device=None) -> Tensor:
    raise NotImplementedError()


def linspace(start, stop, /, num, *, dtype=None, device=None, endpoint=True) -> Tensor:
    raise NotImplementedError()
