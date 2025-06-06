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

from hidet.utils.py import same_list
import hidet.runtime.storage
import hidet.cuda
from hidet.ir import dtypes
from hidet.ir.type import DataType, data_type
from hidet.ir.expr import Expr, symbol_var, is_constant
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
        from hidet.ir.tools import simplify

        self._shape: Tuple[Union[Expr, int], ...] = tuple(simplify(dim, enable_rules=True) for dim in shape)
        self._dtype: DataType = data_type(dtype)
        self._device: Device = instantiate_device(device)
        self._storage: Optional[Storage] = storage
        self._layout: Optional[DataLayout] = layout
        self._trace: Optional[Tuple[Operator, int]] = trace

    @property
    def shape(self) -> Tuple[Union[int, Expr], ...]:
        """
        The shape of the tensor.

        The shape is a tuple of integers indicating the size of the tensor along each dimension.

        Returns
        -------
        shape: Tuple[int, ...]
            The shape of the tensor.
        """
        return self._shape

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
        trace: Tuple[hidet.graph.Operator, int]
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
        layout: Optional[DataLayout]
            The data layout of the tensor. None indicates the compact row major layout.
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
        return prod(dtypes.int64(shape) for shape in self.shape) * self.dtype.nbytes

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

    def __rtruediv__(self, other) -> Tensor:
        from .ops import divide, utils

        return divide(utils.convert_to_tensor(other, self), self)

    def __mod__(self, other) -> Tensor:
        from .ops import mod, utils

        return mod(self, utils.convert_to_tensor(other, self))

    def __pow__(self, power, modulo=None) -> Tensor:
        from .ops import pow, utils

        return pow(self, utils.convert_to_tensor(power, self))

    def __matmul__(self, other) -> Tensor:
        from .ops import utils, matmul

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

    def __ge__(self, other):
        from .ops import greater_equal, utils

        return greater_equal(self, utils.convert_to_tensor(other, self))

    # we do not define __eq__ method for Tensor

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
        from .ops import strided_slice, take
        from .ops import reshape, transpose

        if isinstance(item, Tensor):
            if not item.dtype.is_integer():
                raise TypeError("Tensor indexing via Tensor requires integer index tensor")

            return take(self, item, axis=0)

        if isinstance(item, list):
            item = tuple(item)

        if not isinstance(item, tuple):
            item = tuple([item])

        # process Ellipsis
        # e.g., x[1, ..., 2] -> x[1, :, :, 2]
        if Ellipsis in item:
            if item.count(Ellipsis) > 1:
                raise ValueError('Only one ellipsis allowed in index.')
            ellipsis_index = item.index(Ellipsis)
            ellipsis_ndim = len(self.shape) - sum([1 if axis not in [None, Ellipsis] else 0 for axis in item])
            ellipsis_ndim = max(ellipsis_ndim, 0)
            item = item[:ellipsis_index] + (slice(None),) * ellipsis_ndim + item[ellipsis_index + 1 :]

        # if some elements in item are tensors
        # advanced indexing will be used
        if any(isinstance(it, Tensor) for it in item):
            tensor_indices = []
            slice_indices = []
            for i, it in enumerate(item):
                if isinstance(it, Tensor):
                    tensor_indices.append(i)
                else:
                    slice_indices.append(i)

            if len(self.shape) == 2 and same_list(tensor_indices, list(range(2)), use_equal=True):
                x = self
            else:
                x = transpose(self, tensor_indices + slice_indices + list(range(len(item), len(self.shape))))
            n = len(tensor_indices)

            item_sum = item[tensor_indices[0]] * prod(x.shape[1:n])
            for i in range(1, n):
                item_sum += item[tensor_indices[i]] * prod(x.shape[i + 1 : n])
            x = take(reshape(x, (prod(x.shape[:n]),) + x.shape[n:]), item_sum)

            # check if there is slice index between tensor indices
            # if no, slice indices that came before tensor indices need to go to their initial positions
            # otherwise, they stay as they are at the current line
            if tensor_indices[-1] - tensor_indices[0] + 1 == n:
                transpose_back = []
                new_item = []
                idx = 0
                for _ in range(len(slice_indices)):
                    if slice_indices[idx] == idx:
                        transpose_back.append(len(item[tensor_indices[0]].shape) + idx)
                        new_item.append(item[idx])
                        idx += 1
                    else:
                        break

                for i in range(len(item[tensor_indices[0]].shape)):
                    transpose_back.append(i)
                    new_item.append(slice(None))

                for idx in range(idx, len(slice_indices)):
                    transpose_back.append(idx + len(item[tensor_indices[0]].shape))
                    new_item.append(item[idx + n])

                if len(x.shape) != 2 or not same_list(tensor_indices, list(range(2)), use_equal=True):
                    x = transpose(x, transpose_back)
            else:
                new_item = [slice(None) for _ in range(n)] + [item[idx] for idx in slice_indices]
            return x.__getitem__(new_item)

        # now, the item could have
        # 1. integer index
        # 2. slice
        # 3. None
        # e.g., [1, 3:5, ..., None]

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
                    v = v + self.shape[i]
                if is_constant(v, self.shape[i]) and (v < 0 or v >= self.shape[i]):
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
            if isinstance(v, (int, Expr)):
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
            data = self.detach().cpu().numpy()
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
        r"""
        This function is used to support interoperability with other frameworks that support __dlpack__ protocol.

        The stream specification follows the Python array API standard 2022.
        None: producer must assume the legacy default stream
        1: the legacy default stream
        2: the per-thread default stream
        > 2: stream number represented as a Python integer
        0: is not allowed due to ambiguity: 0 could mean either None, 1, or 2.
        For details, please refer to the following link
        https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__dlpack__.html?highlight=stream

        Notes:
        1. The above convention is specified for CUDA. The specification for ROCm should be added later.
        2. The per-thread default stream is currently not enabled in Hidet. We need to change both the runtime
        code and compilation options to enable it. For details, please refer to the CUDA document
        https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?highlight\=default%20stream\#default-stream
        """
        from .impl.dlpack import to_dlpack

        if stream is not None and self.device.is_cuda():
            if stream == 0:
                raise ValueError(f"Stream({stream}) is not allowed due to its ambiguity.")
            elif stream == 1:
                consumer_stream = hidet.cuda.default_stream()
            elif stream == 2:
                raise NotImplementedError("Currently, Hidet doesn't support per-thread default stream")
            elif stream > 2:
                consumer_stream = hidet.cuda.ExternalStream(stream)
            else:
                raise ValueError(f"Invalid stream number({stream})")
            provider_stream = hidet.cuda.current_stream()
            if consumer_stream != provider_stream:
                event = hidet.cuda.Event()
                event.record(provider_stream)
                consumer_stream.wait_event(event)
        elif stream is not None and self.device.is_hip():
            consumer_stream = hidet.hip.ExternalStream(stream)
            provider_stream = hidet.hip.current_stream()
            if consumer_stream != provider_stream:
                event = hidet.hip.Event()
                event.record(provider_stream)
                consumer_stream.wait_event(event)

        return to_dlpack(self)

    def __dlpack_device__(self) -> Tuple[int, int]:
        """
        This function is used to support interoperability with other frameworks that support __dlpack__ protocol.
        """
        from .impl.dlpack import to_dlpack_device

        return to_dlpack_device(self)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """
        This function is used to support interoperability with PyTorch.

        We can use hidet Tensor as the input of PyTorch function:
        ```
        import torch
        import hidet
        a = hidet.randn([2, 3], dtype='float16', device='cuda')
        b = torch.abs(a)
        ```

        See the following documentation for more information:
        https://pytorch.org/docs/stable/notes/extending.html#extending-torch-with-a-tensor-like-type
        """
        import torch

        if kwargs is None:
            kwargs = {}
        if not all(issubclass(t, (torch.Tensor, Tensor)) for t in types):
            return NotImplemented
        args = (arg.torch() if isinstance(arg, Tensor) else arg for arg in args)
        kwargs = {k: v.torch() if isinstance(v, Tensor) else v for k, v in kwargs.items()}
        return func(*args, **kwargs)

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
            if not isinstance(ret, (int, float, bool, complex)):
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
        if self.layout is None or isinstance(self.layout, RowMajorLayout):
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

    def view(
        self, dtype: Optional[Union[str, DataType]] = None, shape: Optional[List[Union[int, Expr]]] = None
    ) -> Tensor:
        """Reinterpret/view the self tensor with a different `dtype` and/or `shape`,
        without any data copying or casting.

        If the element size of `dtype` is different than `self.dtype`, the size of the last dimension of the new tensor
        will be scaled proportionally. For example, if `dtype.nbytes==2` and `self.dtype.nbytes==4`, the size of the
        last dimension of the new tensor will be doubled. For this to be possible, `len(self.shape)` must be greater
        than 0.

        Additionally, if `dtype.nbytes`>`self.dtype.nbytes`, `self.shape[-1]`
        must be divisible by the ratio between the two element sizes.

        Parameters
        ----------
        dtype: Union[str, DataType], optional
            The target data type to convert to. None indicates unchanged.
        shape: Sequence[int], optional
            The target shape to convert to. None indicates unchanged.

        Returns
        -------
        ret: Tensor
            The tensor with the new data type and shape.

        """
        from .ops import view

        return view(self, dtype, shape)

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
            elif device.is_hip():
                tensor = tensor.hip(device)
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
        from hidet.graph.ops import transfer

        if self.device.kind == 'cpu':
            return self
        else:
            if self.storage is not None:
                return Tensor(self.shape, self.dtype, 'cpu', self.storage.cpu(), self.layout)
            else:
                return transfer(self, 'cpu')

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
        from hidet.graph.ops import transfer

        if device is None:
            device = 'cuda'
        device = instantiate_device(device)
        if self.device == device:
            return self
        else:
            if self.storage is not None:
                return Tensor(self.shape, self.dtype, device, self.storage.cuda(device.id), self.layout)
            else:
                return transfer(self, device)

    def hip(self, device=None):
        """Create a copy of self tensor on hip device.

        If the current tensor is already on hip device, self is returned.

        Parameters
        ----------
        device: Device, optional
            The target hip device. None indicates the current hip device.

        Returns
        -------
        ret: Tensor
            The new tensor or self.
        """
        if device is None:
            device = 'hip'
        device = instantiate_device(device)
        if self.device == device:
            return self
        else:
            if self.storage is not None:
                return Tensor(self.shape, self.dtype, device, self.storage.hip(device.id), self.layout)
            else:
                raise NotImplementedError('Transferring from non-hip device to hip device is not supported yet.')

    def vcuda_(self):
        """Cast the tensor to vcuda device in place.

        If the current tensor is already on vcuda device, nothing is performed

        Returns
        -------
        ret: None
            This operation is in-place
        """

        if self.device.is_vcuda():
            return
        if not self.device.is_cuda():
            raise ValueError("Tensor must be on cuda device, got {}".format(self.device))
        # if the tensor has no storage, there is no need to cast
        if self.storage is not None:
            self._storage = self.storage.vcuda(self.device.id)
        self._device = Device('vcuda', self.device.id)

    def cuda_(self):
        """Cast the tensor from vcuda device in place.

        If the current tensor is already on cuda device, nothing is performed

        Returns
        -------
        ret: None
            This operation is in-place
        """
        if self.device.is_cuda():
            return
        if not self.device.is_vcuda():
            raise ValueError("Tensor must be on vcuda device, got {}".format(self.device))

        if self.storage is not None:
            self._storage = self.storage.cuda(self.device.id)
        self._device = Device('cuda', self.device.id)

    def vhip_(self):
        """Cast the tensor to vhip device in place.

        If the current tensor is already on vhip device, nothing is performed

        Returns
        -------
        ret: None
            This operation is in-place
        """

        if self.device.is_vhip():
            return
        if not self.device.is_hip():
            raise ValueError("Tensor must be on cuda device, got {}".format(self.device))
        # if the tensor has no storage, there is no need to cast
        if self.storage is not None:
            self._storage = self.storage.vhip(self.device.id)
        self._device = Device('vhip', self.device.id)

    def hip_(self):
        """Cast the tensor from vhip device in place.

        If the current tensor is already on cuda device, nothing is performed

        Returns
        -------
        ret: None
            This operation is in-place
        """
        if self.device.is_hip():
            return
        if not self.device.is_vhip():
            raise ValueError("Tensor must be on vhip device, got {}".format(self.device))

        if self.storage is not None:
            self._storage = self.storage.hip(self.device.id)
        self._device = Device('hip', self.device.id)

    def copy(self) -> Tensor:
        """Create a copy of current tensor.

        Returns
        -------
        ret: Tensor
            A new tensor with the same contents as the current one.
        """
        if self.trace is not None:
            raise ValueError('The symbolic tensor is not modifiable, so feel free to use them without copying.')
        return Tensor(
            shape=list(self.shape),
            dtype=self.dtype,
            device=self.device,
            storage=self.storage.copy(),
            layout=self.layout,
            trace=None,
        )

    def copy_(self, src: Tensor):
        src_converted = src.to(dtype=self.dtype, device=self.device)
        if len(src.shape) != len(self.shape) or any(a != b for a, b in zip(self.shape, src_converted.shape)):
            raise ValueError(
                'The shape of source tensor {} does not match the shape of target tensor {}'.format(
                    src_converted.shape, self.shape
                )
            )
        if src_converted is src:
            self._storage = src.storage.copy()
        else:
            self._storage = src.storage
        return self

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
            raise ValueError('The symbolic tensor is not modifiable, so feel free to use them without copying.')
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
        if self.device.kind == 'cpu':
            return self
        else:
            if self.trace is None:
                ret = Tensor(
                    self.shape, self.dtype, 'cpu', self.storage.cpu_async(stream) if self.storage else None, self.layout
                )
                return ret
            else:
                raise ValueError('Please use .cpu() for symbolic tensor transfer.')

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
                raise ValueError('Please use .cuda(...) for symbolic tensor transfer.')

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
        if self.device.kind != 'cpu':
            raise RuntimeError('Cannot convert a tensor on {} to numpy array.'.format(self.device))
        if self.dtype in [dtypes.bfloat16, dtypes.tfloat32, dtypes.float8_e4m3, dtypes.float8_e5m2]:
            warnings.warn('numpy does not support {}, converting to float32'.format(self.dtype.name))
            return self.astype(dtypes.float32).numpy()
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
        from .frontend.torch.utils import dtype_to_torch

        if self.dtype in [hidet.float8_e4m3, hidet.float8_e5m2]:
            old_dtype = self.dtype
            self._dtype = hidet.uint8
            res = self.torch().view(dtype_to_torch(old_dtype))
            self._dtype = old_dtype
            return res

        return torch.from_dlpack(self)

    def masked_fill(self, mask, value):
        """
        Fills the tensor with value where mask is True

        Parameters
        ----------
        mask: Tensor
            The target cuda device. None indicates the current cuda device.

        value: Union[float, int]
            The stream to copy the tensor to GPU on. None indicates the current stream.

        Returns
        -------
        ret: Tensor
        """
        from .ops import where

        return where(mask, full([], value, dtype=self.dtype, device=self.device), self)

    def expand(self, *sizes: int) -> Tensor:
        from .ops import broadcast

        sizes: List[int] = list(sizes)
        assert len(sizes) >= len(self.shape)
        for i in range(len(sizes)):
            if sizes[i] == -1:
                ri = len(sizes) - 1 - i
                assert ri < len(self.shape)
                sizes[i] = int(self.shape[len(self.shape) - 1 - ri])
        return broadcast(self, sizes)

    def float(self) -> Tensor:
        return self.to(dtype=hidet.float32)

    def transpose(self, dim0: int, dim1: int):
        from .ops import transpose

        if dim0 < dim1:
            dim0, dim1 = dim1, dim0
        return transpose(self, [dim0, dim1])


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
    num_bytes = int(prod(shape) * dtype.nbits // 8)
    storage = Storage.new(device, num_bytes)
    return Tensor(shape=shape, dtype=dtype, device=device, storage=storage, layout=layout)


def symbol(shape: Sequence[Union[int, str, Expr]], dtype='float32', device='cpu', layout=None) -> Tensor:
    """Create a symbolic tensor.

    Parameters
    ----------
    shape: Sequence[Union[int, str, Expr]]
        The shape of new tensor. The shape can contain symbolic variables. str indicates the corresponding dimension is
        a symbolic variable with the given name.

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
    updated_shape = []
    for d in shape:
        if isinstance(d, str):
            updated_shape.append(symbol_var(d))
        else:
            updated_shape.append(d)
    return Tensor(shape=updated_shape, dtype=dtype, device=device, storage=None, layout=layout)


def zeros(shape: Sequence[int], dtype: Union[DataType, str] = 'float32', device='cpu') -> Tensor:
    """Create a tensor initialized with zero.

    Parameters
    ----------
    shape: Sequence[int]
        The shape of new tensor.

    dtype: str or DataType
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
    """Create a tensor with normal (Gaussian) distributed values.

    Parameters
    ----------
    shape: Sequence[int]
        The shape of new tensor.

    dtype: DataType or str, default 'float32'
        The data type of element of the tensor.

    mean: float, default 0.0
        The mean of the normal distribution.

    stddev: float, default 1.0
        The standard deviation of the normal distribution.

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
    dtype: DataType = data_type(dtype)
    if dtype.is_complex():
        assert isinstance(dtype, dtypes.complex.ComplexType)
        real = hidet.randn(shape, dtype=dtype.base_dtype, mean=mean, stddev=stddev, device=device)
        imag = hidet.randn(shape, dtype=dtype.base_dtype, mean=mean, stddev=stddev, device=device)
        return real + imag * 1j
    else:
        if any(not isinstance(d, int) for d in shape):
            raise RuntimeError('shape must be a sequence of integers, got {}'.format(repr(shape)))
        np_tensor = np.array(np.random.randn(*shape) * stddev + mean).astype(
            np.float32
        )  # wrap np.array(...) for shape=[]
        dtype = data_type(dtype)

        if isinstance(np_tensor, float):  # shape = []
            np_tensor = np.array(np_tensor)

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
    dtype: Optional[Union[str, DataType]] = None,
    device: Optional[str] = None,
) -> Tensor:
    """
    Create a tensor initialized with fill_value with the same shape, dtype, and device as the given tensor.

    Parameters
    ----------
    data: Tensor
        The tensor to copy shape, dtype, and device from.

    fill_value: int, float, bool, complex
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
    data: Tensor,
    mean: float = 0.0,
    stddev: float = 1.0,
    shape: Optional[Sequence[int]] = None,
    dtype: Optional[str] = None,
    device: Optional[str] = None,
) -> Tensor:
    """
    Create a randomly initialized tensor with the same shape, dtype, and device as the given tensor.

    Parameters
    ----------
    data: Tensor
        The tensor to copy shape, dtype, and device from.

    mean: float, optional
        The mean of the normal distribution.

    stddev: float, optional
        The standard deviation of the normal distribution.

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
        mean=mean,
        stddev=stddev,
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
    from .frontend.torch.utils import dtype_from_torch

    if not isinstance(torch_tensor, torch.Tensor):
        raise ValueError('Expect a torch.Tensor, got {}'.format(type(torch_tensor)))
    if torch_tensor.requires_grad:
        torch_tensor = torch_tensor.detach()
    if torch_tensor.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:  #
        return from_torch(torch_tensor.view(torch.uint8)).view(dtype_from_torch(torch_tensor.dtype))
    return from_dlpack(torch_tensor)


def asarray(obj, /, *, dtype=None, device=None) -> Tensor:
    """
    Convert a list, tuple, or numpy ndarray to a hidet tensor.

    Parameters
    ----------
    obj: bool, int, float, List, Tuple, Tensor, np.ndarray
        The object to be converted.

    dtype: DataType or str, optional
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
        array = np.array(obj, dtype=dtype_to_numpy(data_type(dtype)) if dtype else None)
        if array.dtype == np.float64:
            # numpy uses float64 as the default float data type, convert it to float32 as hidet takes float32 as default
            array = array.astype(np.float32)
        ret = from_numpy(array)
    return ret.to(dtype=dtype, device=device)
