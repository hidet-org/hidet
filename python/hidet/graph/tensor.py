from __future__ import annotations

import ctypes
from functools import partial
from typing import List, Optional, Tuple, Sequence, Union
import warnings

import numpy as np

import hidet.runtime.storage
from hidet.ffi import cuda, cuda_kernels
from hidet.ir import dtypes
from hidet.ir.type import DataType, data_type
from hidet.ir.layout import DataLayout, RowMajorLayout
from hidet.runtime.cuda_stream import CudaStream
from hidet.runtime.storage import Storage
from hidet.utils import prod


def convert(v, device: str):
    if isinstance(v, (float, int)):
        dtype_map = {float: 'float32', int: 'int64'}
        return full(shape=[1], fill_value=v, dtype=dtype_map[type(v)], device=device)
    elif isinstance(v, Tensor):
        return v
    else:
        raise NotImplementedError()


class Tensor:
    """An n-dimension array, could be symbolic or concrete.

    This class defines an n-dimension array.

    Attributes
    ----------
    shape: List[int]
        The shape of the tensor.

    dtype: DataType
        The data type of the tensor.

    device: str
        The device of the tensor.

    layout: DataLayout
        The data layout of the tensor.

    storage: Optional[Storage]
        The storage of the tensor. None indicates it is a symbolic tensor.

    trace: Optional[Tuple[Operator, int]]
        Where this tensor is derived from. A trace = (op, i) indicates that this tensor is the i-th output of the op
        operator.
    """

    def __init__(
        self,
        shape: Sequence[int],
        dtype: Union[str, DataType],
        device: str,
        storage: Optional[Storage],
        layout: Optional[DataLayout] = None,
        trace: Optional[Tuple['Operator', int]] = None,
    ):
        from hidet.graph.operator import Operator

        self.shape: List[int] = [int(v) for v in shape]
        self.dtype: DataType = data_type(dtype)
        self.device: str = device
        self.storage: Optional[Storage] = storage
        self.layout: DataLayout = layout if layout else DataLayout.row_major(shape)
        self.trace: Optional[Tuple[Operator, int]] = trace

    def __neg__(self) -> Tensor:
        from .ops import neg

        return neg(self)

    def __add__(self, other) -> Tensor:
        from .ops import add

        return add(self, other)

    def __radd__(self, other):
        from .ops import add

        return add(other, self)

    def __sub__(self, other) -> Tensor:
        from .ops import sub

        return sub(self, other)

    def __rsub__(self, other):
        from .ops import sub

        return sub(other, self)

    def __mul__(self, other) -> Tensor:
        from .ops import multiply

        return multiply(self, other)

    def __rmul__(self, other):
        from .ops import multiply

        return multiply(other, self)

    def __truediv__(self, other) -> Tensor:
        from .ops import divide

        return divide(self, other)

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

    def __iter__(self):
        raise TypeError('hidet.Tensor does not support iteration.')

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

        if stream:
            cuda.stream_synchronize(stream)
        return to_dlpack(self)

    def __dlpack_device__(self) -> Tuple[int, int]:
        """
        This function is used to support interoperability with other frameworks that support __dlpack__ protocol.
        """
        from .impl.dlpack import to_dlpack_device

        return to_dlpack_device(self)

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
        ret: Optional[hidet.graph.operator.Operator]
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
        value = self.numpy().tolist()
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

    def to(self, dtype: Optional[str] = None, device: Optional[str] = None):
        """Cast the data type of current tensor or move it to another device.

        Parameters
        ----------
        dtype: Optional[str]
            The target data type to convert to. None indicates unchanged.

        device: Optional[str]
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
            if device == 'cpu':
                tensor = tensor.cpu()
            elif device == 'cuda':
                tensor = tensor.cuda()
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
        if self.device == 'cpu':
            return self
        else:
            if self.trace is None:
                return Tensor(self.shape, self.dtype, 'cpu', self.storage.cpu() if self.storage else None, self.layout)
            else:
                raise ValueError('Please use .detach() to detach a trace variable first.')

    def cuda(self):
        """Create a copy of self tensor on cuda device.

        If the current tensor is already on cuda device, self is returned.

        Returns
        -------
        ret: Tensor
            The new tensor or self.
        """
        if self.device == 'cuda':
            return self
        else:
            if self.trace is None:
                return Tensor(
                    self.shape, self.dtype, 'cuda', self.storage.cuda() if self.storage else None, self.layout
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

    def copy_async(self, stream: Optional[CudaStream] = None) -> Tensor:
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

    def cpu_async(self, stream: Optional[CudaStream] = None):
        """
        Copy the tensor to CPU asynchronously.

        Parameters
        ----------
        stream: Optional[CudaStream]
            The stream to copy the tensor to CPU on.

        Returns
        -------
        ret: Tensor
            The tensor on CPU.
        """
        if self.device == 'cpu':
            return self
        else:
            if self.trace is None:
                ret = Tensor(
                    self.shape, self.dtype, 'cpu', self.storage.cpu_async(stream) if self.storage else None, self.layout
                )
                return ret
            else:
                raise ValueError('Please use .detach() to detach a trace variable first.')

    def cuda_async(self, stream: Optional[CudaStream] = None):
        """
        Copy the tensor to GPU asynchronously.

        Parameters
        ----------
        stream: Optional[CudaStream]
            The stream to copy the tensor to GPU on.

        Returns
        -------
        ret: Tensor
            The tensor on GPU.
        """
        if self.device == 'cuda':
            return self
        else:
            if self.trace is None:
                ret = Tensor(
                    self.shape,
                    self.dtype,
                    'cuda',
                    self.storage.cuda_async(stream) if self.storage else None,
                    self.layout,
                )
                return ret
            else:
                raise ValueError('Please use .detach() to detach a trace variable first.')

    def numpy(self, share_mem=True) -> np.ndarray:
        """
        Convert the tensor to a numpy array.

        If this is a cpu tensor and share_mem is True, the numpy array will share the memory with the tensor when
        possible.

        Parameters
        ----------
        share_mem: bool
            Whether to share the memory with the tensor when possible.

        Returns
        -------
        ret: np.ndarray
            The numpy array.
        """
        if self.device != 'cpu':
            return self.cpu().numpy()
        if self.dtype in [dtypes.bfloat16, dtypes.tfloat32]:
            warnings.warn('numpy does not support {}, converting to float32'.format(self.dtype.name))
            return self.cast(dtypes.float32).numpy()

        storage = self.contiguous().storage  # convert if this tensor is not in row major layout
        np_array = storage.as_array(num_elements=prod(self.shape), dtype=self.dtype, share_mem=share_mem)
        np_array: np.ndarray = np_array.reshape(self.shape)
        return np_array

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


def empty(shape, dtype: str = 'float32', device: str = 'cuda', layout: Optional[DataLayout] = None):
    """Create an uninitialized tensor.

    Parameters
    ----------
    shape: Sequence[int]
        The shape of new tensor.

    dtype: str
        The data type of element of the tensor.

    device: str
        The device of the new tensor is created on.

    layout: Optional[DataLayout]
        The data layout of the tensor.

    Returns
    -------
    ret: Tensor
        The created tensor.
    """
    num_bytes = prod(shape) * data_type(dtype).nbytes
    storage = Storage.new(device, num_bytes)
    return Tensor(shape, dtype, device, storage, layout)


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

    layout: Optional[DataLayout]
        The data layout of the tensor.

    Returns
    -------
    ret: Tensor
        The created tensor.

    """
    return Tensor(shape, dtype, device, None, layout)


def zeros(shape: Sequence[int], dtype='float32', device='cuda', layout=None) -> Tensor:
    """Create a tensor initialized with zero.

    Parameters
    ----------
    shape: Sequence[int]
        The shape of new tensor.

    dtype: str
        The data type of element of the tensor.

    device: str
        The device of the new tensor is created on.

    layout: Optional[DataLayout]
        The data layout of the tensor.

    Returns
    -------
    ret: Tensor
        The created tensor.
    """
    tensor = empty(shape, dtype, device, layout)
    cuda.memset_async(tensor.storage.addr, tensor.nbytes, value=0)
    return tensor


def ones(shape, dtype='float32', device='cuda', layout=None) -> Tensor:
    """Create a tensor initialized with one.

    Parameters
    ----------
    shape: Sequence[int]
        The shape of new tensor.

    dtype: str
        The data type of element of the tensor.

    device: str
        The device of the new tensor is created on.

    layout: Optional[DataLayout]
        The data layout of the tensor.

    Returns
    -------
    ret: Tensor
        The created tensor.
    """
    value_map = {'float32': 1.0, 'int32': 1, 'int64': 1}
    if dtype in value_map:
        return full(shape, value_map[dtype], dtype, device, layout)
    else:
        if dtype in ['float16', 'bool']:
            f32_tensor = ones(shape, 'float32', device, layout)
            return f32_tensor.cast(dtype)
        else:
            raise NotImplementedError(
                'Not implemented ones for dtype {}, please create a float32 tensor and cast to this type'.format(dtype)
            )


def full(shape, fill_value: Union[float, int], dtype='float32', device='cuda', layout=None) -> Tensor:
    """Create a tensor initialized with given constant.

    Parameters
    ----------
    shape: Sequence[int]
        The shape of new tensor.

    fill_value: Union[float, int]
        The constant to initialize the new tensor.

    dtype: DataType or str
        The data type of element of the tensor.

    device: str
        The device of the new tensor is created on.

    layout: DataLayout or None
        The data layout of the tensor.

    Returns
    -------
    ret: Tensor
        The created tensor.
    """
    tensor = empty(shape, dtype, device, layout)
    cuda_kernels.fill_value(tensor.storage.addr, num_elements=tensor.num_elements, value=fill_value, dtype=dtype)
    return tensor


def randn(shape, dtype='float32', mean=0.0, stddev=1.0, device='cuda', layout=None) -> Tensor:
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

    layout: DataLayout or None
        The data layout of the tensor.

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
    if device != 'cuda' or dtype != 'float32':
        return randn(shape, 'float32', mean, stddev, 'cuda', layout).to(device=device, dtype=dtype)

    assert device == 'cuda' and dtype == 'float32'
    tensor = empty(shape, dtype='float32', device='cuda', layout=layout)
    cuda.generate_normal(tensor.storage.addr, num_elements=prod(tensor.shape), mean=mean, stddev=stddev)
    return tensor


def randint(low: int, high=None, shape: Sequence[int] = (), dtype: str = 'int32') -> Tensor:
    dtype_map = {'int32': np.int32, 'int64': np.int64}
    if dtype not in dtype_map:
        return randint(low=low, high=high, shape=shape, dtype='int32').cast(dtype)
    return array(np.random.randint(low=low, high=high, size=shape, dtype=dtype_map[dtype]))


def _tensor_like(constructor, data: Tensor, shape, dtype, device, layout):
    shape = data.shape if shape is None else shape
    dtype = data.dtype if dtype is None else dtype
    device = data.device if device is None else device
    layout = data.layout if layout is None else layout
    return constructor(shape=shape, dtype=dtype, device=device, layout=layout)


def empty_like(
    data: Tensor,
    shape: Optional[Sequence[int]] = None,
    dtype: Optional[str] = None,
    device: Optional[str] = None,
    layout: Optional[DataLayout] = None,
) -> Tensor:
    return _tensor_like(empty, data, shape, dtype, device, layout)


def symbol_like(data: Tensor, shape=None, dtype=None, device=None, layout=None):
    """Create a symbol tensor like an existing tensor.

    Parameters
    ----------
    data: Tensor
        The information of this tensor will be used to create the symbol tensor.

    shape: Optional[Sequence[int]]
        The shape of the new tensor.

    dtype: Optional[str]
        The data type of the new tensor.

    device: Optional[str]
        The device of the new tensor.

    layout: Optional[DataLayout]
        The data layout of the new tensor.

    Returns
    -------
    ret: Tensor
        The created symbol tensor.
    """
    return _tensor_like(symbol, data, shape, dtype, device, layout)


def zeros_like(
    data: Tensor,
    shape: Optional[Sequence[int]] = None,
    dtype: Optional[str] = None,
    device: Optional[str] = None,
    layout: Optional[DataLayout] = None,
) -> Tensor:
    return _tensor_like(zeros, data, shape, dtype, device, layout)


def ones_like(
    data: Tensor,
    shape: Optional[Sequence[int]] = None,
    dtype: Optional[str] = None,
    device: Optional[str] = None,
    layout: Optional[DataLayout] = None,
) -> Tensor:
    return _tensor_like(ones, data, shape, dtype, device, layout)


def full_like(
    data: Tensor,
    fill_value,
    shape: Optional[Sequence[int]] = None,
    dtype: Optional[str] = None,
    device: Optional[str] = None,
    layout: Optional[DataLayout] = None,
) -> Tensor:
    return _tensor_like(partial(full, fill_value=fill_value), data, shape, dtype, device, layout)


def randn_like(
    data: Tensor,
    shape: Optional[Sequence[int]] = None,
    dtype: Optional[str] = None,
    device: Optional[str] = None,
    layout: Optional[DataLayout] = None,
) -> Tensor:
    return _tensor_like(randn, data, shape, dtype, device, layout)


def randint_like(
    data: Tensor,
    low: int,
    high: Optional[int] = None,
    shape: Optional[Sequence[int]] = None,
    dtype: Optional[str] = None,
    device: Optional[str] = None,
    layout: Optional[DataLayout] = None,
):
    return _tensor_like(partial(randint, low=low, high=high), data, shape, dtype, device, layout)


def void_pointer_to_uint64(p):
    ret = ctypes.cast(ctypes.addressof(p), ctypes.POINTER(ctypes.c_uint64)).contents
    return ret.value


def from_numpy(nparray: np.ndarray) -> Tensor:
    dtype_convert = {
        np.dtype(np.float64): 'float64',
        np.dtype(np.float32): 'float32',
        np.dtype(np.int64): 'int64',
        np.dtype(np.int32): 'int32',
        np.dtype(np.float16): 'float16',
        np.dtype(np.bool): 'bool',
        np.dtype(np.uint8): 'uint8',
        np.dtype(np.uint32): 'uint32',
    }
    if nparray.dtype not in dtype_convert:
        raise NotImplementedError("Do not support convert np.ndarray with data type '{}'.".format(nparray.dtype))
    nparray = nparray.copy(order='C')  # make the data layout like C, which is contiguous
    tensor = empty(shape=nparray.shape, dtype=dtype_convert[nparray.dtype], device='cpu')
    cuda.memcpy(
        src_addr=void_pointer_to_uint64(nparray.ctypes.data_as(ctypes.c_void_p)),
        dst_addr=tensor.storage.addr,
        num_bytes=tensor.nbytes,
        kind=cuda.HostToHost,
    )
    return tensor


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
    return hidet.from_dlpack(torch_tensor)


def array(obj: Union[float, int, List, Tuple, np.ndarray, Tensor]) -> Tensor:
    """Convert a list, tuple, or numpy ndarray to a hidet tensor.

    Parameters
    ----------
    obj: Union[List, Tuple, np.ndarray, Tensor]
        The object to be converted.

    Returns
    -------
    ret: Tensor
        The hidet tensor converted from given object.
    """
    if isinstance(obj, np.ndarray):
        return from_numpy(obj)
    elif isinstance(obj, Tensor):
        return obj
    else:
        return from_numpy(np.array(obj))
