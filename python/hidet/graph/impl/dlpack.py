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
from typing import List, Callable, Set, Tuple
import ctypes
from ctypes import pythonapi
import hidet.ir
from hidet.ir import DataType, dtypes
from hidet.utils import initialize, prod
from hidet.runtime.device import Device
from hidet.runtime.storage import Storage
from hidet.ir.type import data_type
from hidet.graph.tensor import Tensor


class DLDeviceType(ctypes.c_int32):
    kDLCPU = 1
    kDLGPU = 2
    kDLCPUPinned = 3
    kDLOpenCL = 4
    kDLVulkan = 7
    kDLMetal = 8
    kDLVPI = 9
    kDLROCM = 10
    kDLOpenGL = 11
    kDLExtDev = 12


class DLDataTypeCode(ctypes.c_int8):
    kDLInt = 0
    kDLUInt = 1
    kDLFloat = 2
    kDLOpaqueHandle = 3
    kDLBfloat = 4
    kDLComplex = 5
    kDLBool = 6


DLTensorDeleter = ctypes.CFUNCTYPE(None, ctypes.c_void_p)


class DLDevice(ctypes.Structure):
    _fields_ = [('device_type', ctypes.c_int32), ('device_id', ctypes.c_int32)]


class DLDataType(ctypes.Structure):
    _fields_ = [('code', ctypes.c_int8), ('bits', ctypes.c_int8), ('lanes', ctypes.c_int16)]
    _dtype_map = {
        dtypes.int8: (DLDataTypeCode.kDLInt, 8, 1),
        dtypes.int16: (DLDataTypeCode.kDLInt, 16, 1),
        dtypes.int32: (DLDataTypeCode.kDLInt, 32, 1),
        dtypes.int64: (DLDataTypeCode.kDLInt, 64, 1),
        dtypes.uint8: (DLDataTypeCode.kDLUInt, 8, 1),
        dtypes.uint16: (DLDataTypeCode.kDLUInt, 16, 1),
        dtypes.uint32: (DLDataTypeCode.kDLUInt, 32, 1),
        dtypes.uint64: (DLDataTypeCode.kDLUInt, 64, 1),
        dtypes.float16: (DLDataTypeCode.kDLFloat, 16, 1),
        dtypes.float32: (DLDataTypeCode.kDLFloat, 32, 1),
        dtypes.float64: (DLDataTypeCode.kDLFloat, 64, 1),
        dtypes.bfloat16: (DLDataTypeCode.kDLBfloat, 16, 1),
        dtypes.boolean: (DLDataTypeCode.kDLBool, 8, 1),
    }

    @staticmethod
    def from_dtype(dtype: DataType) -> DLDataType:
        return DLDataType._dtype_map[dtype]

    @staticmethod
    def to_dtype(dl_dtype: DLDataType) -> DataType:
        dtype_code: int = dl_dtype.code
        dtype_bits: int = dl_dtype.bits
        dtype_lanes: int = dl_dtype.lanes
        if dtype_lanes > 1:
            raise ValueError('from_dlpack: only tensors with lanes=1 are supported for hidet')
        if dtype_code == DLDataTypeCode.kDLInt:
            dtype_name = 'int'
        elif dtype_code == DLDataTypeCode.kDLUInt:
            dtype_name = 'uint'
        elif dtype_code == DLDataTypeCode.kDLFloat:
            dtype_name = 'float'
        elif dtype_code == DLDataTypeCode.kDLOpaqueHandle:
            raise ValueError('from_dlpack: opaque handle is not supported for hidet')
        elif dtype_code == DLDataTypeCode.kDLBfloat:
            dtype_name = 'bfloat'
        elif dtype_code == DLDataTypeCode.kDLComplex:
            raise ValueError('from_dlpack: complex is not supported for hidet')
        elif dtype_code == DLDataTypeCode.kDLBool:
            dtype_name = 'bool'
        else:
            raise ValueError('from_dlpack: unknown dtype code {}'.format(dtype_code))
        if dtype_name != 'bool':
            dtype = '{}{}'.format(dtype_name, dtype_bits)
        else:
            if dtype_bits != 8:
                raise ValueError('from_dlpack: hidet only supports bool dtype with 8 bits.')
            dtype = dtype_name
        if not hidet.ir.dtypes.supported(dtype):
            raise ValueError('from_dlpack: dtype {} is not supported for hidet'.format(dtype))
        return data_type(dtype)


class DLTensor(ctypes.Structure):
    _fields_ = [
        ('data', ctypes.c_void_p),
        ('device', DLDevice),
        ('ndim', ctypes.c_int32),
        ('dtype', DLDataType),
        ('shape', ctypes.POINTER(ctypes.c_int64)),
        ('strides', ctypes.POINTER(ctypes.c_int64)),
        ('byte_offset', ctypes.c_uint64),
    ]


class DLManagedTensor(ctypes.Structure):
    _fields_ = [('dl_tensor', DLTensor), ('manager_ctx', ctypes.c_uint64), ('deleter', DLTensorDeleter)]


@initialize()
def initialize_pythonapi_restype():
    pythonapi.PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
    pythonapi.PyCapsule_New.restype = ctypes.py_object
    pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
    pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
    pythonapi.PyCapsule_IsValid.argtypes = [ctypes.py_object, ctypes.c_char_p]
    pythonapi.PyCapsule_IsValid.restype = ctypes.c_int32
    pythonapi.PyCapsule_SetName.argtypes = [ctypes.py_object, ctypes.c_char_p]
    pythonapi.PyCapsule_SetName.restype = ctypes.c_int32


class DLPackStorage(Storage):
    def __init__(
        self, device: Device, addr: int, num_bytes: int, managed_tensor_addr: int, deleter: Callable[[int], None]
    ):
        super().__init__(
            device=device, addr=addr, num_bytes=num_bytes, free_handler=lambda storage: deleter(managed_tensor_addr)
        )


def is_compact_tensor(shape: List[int], strides: List[int]) -> bool:
    assert len(shape) == len(strides)
    current_stride = 1
    for extent, stride in zip(reversed(shape), reversed(strides)):
        if stride != current_stride and extent != 1:
            return False
        current_stride *= extent
    return True


def read_longs(ptr: ctypes.POINTER(ctypes.c_int64), length: int) -> List[int]:
    array_ptr = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_int64 * length))
    return [int(v) for v in array_ptr.contents]


def from_dlpack_capsule(dltensor) -> Tensor:
    if pythonapi.PyCapsule_IsValid(dltensor, b'dltensor'):
        ptr: int = pythonapi.PyCapsule_GetPointer(dltensor, b'dltensor')
        managed_tensor: DLManagedTensor = ctypes.cast(ptr, ctypes.POINTER(DLManagedTensor)).contents
        tensor = managed_tensor.dl_tensor

        # shape
        shape: List[int] = read_longs(tensor.shape, tensor.ndim)

        # dtype
        dtype: DataType = DLDataType.to_dtype(tensor.dtype)

        # device
        device_type: str = tensor.device.device_type
        device_id: int = tensor.device.device_id
        if device_type in [DLDeviceType.kDLCPU, DLDeviceType.kDLCPUPinned]:
            device = Device('cpu')
            assert device_id == 0
        elif device_type == DLDeviceType.kDLGPU:
            device = Device('cuda', device_id)
        else:
            raise ValueError('from_dlpack: currently, hidet only supports only CPU and GPU tensors.')

        # storage
        if tensor.strides:
            # tensor.strides != nullptr, we need to check it is a compact tensor in row-major layout
            strides: List[int] = read_longs(tensor.strides, tensor.ndim)
            if not is_compact_tensor(shape, strides):
                raise ValueError(
                    (
                        'from_dlpack: got tensor with shape {} and strides {}. Only compact tensors are supported for '
                        'hidet, please consider make it continuous before passing it to hidet.'
                    ).format(shape, strides)
                )
        else:
            # tensor.strides = nullptr, a compact tensor
            pass
        storage = DLPackStorage(
            device=device,
            addr=tensor.data + tensor.byte_offset,
            num_bytes=dtype.nbytes * prod(shape),
            managed_tensor_addr=ptr,
            deleter=managed_tensor.deleter,
        )

        # mark the name of the capsule to be 'used_dltensor' to prevent double usage
        pythonapi.PyCapsule_SetName(dltensor, b'used_dltensor')

        return Tensor(shape=shape, dtype=dtype, device=device, storage=storage)

    raise RuntimeError('Expect a dltensor in PyCapsule and not been consumed before.')


class DLManagedTensorContext:
    allocated: Set[DLManagedTensorContext] = set()

    def __init__(self, tensor: Tensor):
        ndim = len(tensor.shape)
        self.shape = (ctypes.c_uint64 * ndim)(*tensor.shape)

        self.tensor = tensor
        if tensor.device.is_cuda():
            dl_device = DLDevice(DLDeviceType.kDLGPU, tensor.device.id)
        else:
            dl_device = DLDevice(DLDeviceType.kDLCPU, 0)
        dl_tensor = DLTensor(
            data=tensor.storage.addr,
            device=dl_device,
            ndim=ndim,
            dtype=DLDataType.from_dtype(tensor.dtype),
            shape=ctypes.cast(self.shape, ctypes.POINTER(ctypes.c_int64)),
            strides=None,
            byte_offset=0,
        )
        self.allocated.add(self)

        def deleter(dl_managed_tensor_addr: int):
            _ = dl_managed_tensor_addr  # unused
            self.tensor = None  # free the tensor
            self.allocated.remove(self)

        self.managed_tensor = DLManagedTensor(dl_tensor=dl_tensor, manager_ctx=0, deleter=DLTensorDeleter(deleter))

    def capsuled_dltensor(self) -> ctypes.py_object:
        return pythonapi.PyCapsule_New(ctypes.byref(self.managed_tensor), b'dltensor', None)


def to_dlpack(tensor: Tensor) -> ctypes.py_object:
    return DLManagedTensorContext(tensor).capsuled_dltensor()


def to_dlpack_device(tensor: Tensor) -> Tuple[int, int]:
    if tensor.device.type == 'cuda':
        return DLDeviceType.kDLGPU, 0
    elif tensor.device.type == 'cpu':
        # here, we use kDLCPU instead of kDLCPUPinned, because we pytorch doesn't support kDLCPUPinned
        # technically, we can think kDLCPUPinned as a special case of kDLCPU
        device_type = DLDeviceType.kDLCPU
        return device_type, 0
    else:
        raise NotImplementedError()
