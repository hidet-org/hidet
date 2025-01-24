import ctypes
from ctypes import Structure, c_uint, c_int, c_char, c_void_p, c_size_t, POINTER, byref
from typing import Tuple


class FakeObj:
    def __getattr__(self, item):
        return FakeObj()

    def __setattr__(self, key, value):
        pass


# libamdhip = ctypes.cdll.LoadLibrary("libamdhip64.so")
libamdhip = FakeObj()


# Define C structures used by the libamdhip APIs
class hipDeviceArch_t(Structure):
    _fields_ = [
        ('hasGlobalInt32Atomics', c_uint),
        ('hasGlobalFloatAtomicExch', c_uint),
        ('hasSharedInt32Atomics', c_uint),
        ('hasSharedFloatAtomicExch', c_uint),
        ('hasFloatAtomicAdd', c_uint),
        ('hasGlobalInt64Atomics', c_uint),
        ('hasSharedInt64Atomics', c_uint),
        ('hasDoubles', c_uint),
        ('hasWarpVote', c_uint),
        ('hasWarpBallot', c_uint),
        ('hasWarpShuffle', c_uint),
        ('hasFunnelShift', c_uint),
        ('hasThreadFenceSystem', c_uint),
        ('hasSyncThreadsExt', c_uint),
        ('hasSurfaceFuncs', c_uint),
        ('has3dGrid', c_uint),
        ('hasDynamicParallelism', c_uint),
    ]


class hipDeviceProp_t(Structure):
    _fields_ = [
        ('name', c_char * 256),
        ('totalGlobalMem', c_size_t),
        ('sharedMemPerBlock', c_size_t),
        ('regsPerBlock', c_int),
        ('warpSize', c_int),
        ('maxThreadsPerBlock', c_int),
        ('maxThreadsDim', c_int * 3),
        ('maxGridSize', c_int * 3),
        ('clockRate', c_int),
        ('memoryClockRate', c_int),
        ('memoryBusWidth', c_int),
        ('totalConstMem', c_size_t),
        ('major', c_int),
        ('minor', c_int),
        ('multiProcessorCount', c_int),
        ('l2CacheSize', c_int),
        ('maxThreadsPerMultiProcessor', c_int),
        ('computeMode', c_int),
        ('clockInstructionRate', c_int),
        ('arch', hipDeviceArch_t),
        ('concurrentKernels', c_int),
        ('pciDomainID', c_int),
        ('pciBusID', c_int),
        ('pciDeviceID', c_int),
        ('maxSharedMemoryPerMultiProcessor', c_size_t),
        ('isMultiGpuBoard', c_int),
        ('canMapHostMemory', c_int),
        ('gcnArch', c_int),
        ('gcnArchName', c_char * 256),
        ('integrated', c_int),
        ('cooperativeLaunch', c_int),
        ('cooperativeMultiDeviceLaunch', c_int),
        ('maxTexture1DLinear', c_int),
        ('maxTexture1D', c_int),
        ('maxTexture2D', c_int * 2),
        ('maxTexture3D', c_int * 3),
        ('hdpMemFlushCntl', POINTER(c_uint)),
        ('hdpRegFlushCntl', POINTER(c_uint)),
        ('memPitch', c_size_t),
        ('textureAlignment', c_size_t),
        ('texturePitchAlignment', c_size_t),
        ('kernelExecTimeoutEnabled', c_int),
        ('ECCEnabled', c_int),
        ('tccDriver', c_int),
        ('cooperativeMultiDeviceUnmatchedFunc', c_int),
        ('cooperativeMultiDeviceUnmatchedGridDim', c_int),
        ('cooperativeMultiDeviceUnmatchedBlockDim', c_int),
        ('cooperativeMultiDeviceUnmatchedSharedMem', c_int),
        ('isLargeBar', c_int),
        ('asicRevision', c_int),
        ('managedMemory', c_int),
        ('directManagedMemAccessFromHost', c_int),
        ('concurrentManagedAccess', c_int),
        ('pageableMemoryAccess', c_int),
        ('pageableMemoryAccessUsesHostPageTables', c_int),
    ]


# Define argtypes and restypes for each libamdhip API
_hipGetErrorString = libamdhip.hipGetErrorString
_hipGetErrorString.argtypes = [ctypes.c_int]
_hipGetErrorString.restype = ctypes.c_char_p

_hipMemGetInfo = libamdhip.hipMemGetInfo
_hipMemGetInfo.argtypes = [POINTER(c_size_t), POINTER(c_size_t)]
_hipMemGetInfo.restype = c_int  # Return type is technically hipError_t, which is an enum where hipSuccess is 0

_hipMalloc = libamdhip.hipMalloc
_hipMalloc.argtypes = [POINTER(c_void_p), c_size_t]
_hipMalloc.restype = c_int

_hipFree = libamdhip.hipFree
_hipFree.argtypes = [c_void_p]
_hipFree.restype = c_int

_hipHostMalloc = libamdhip.hipHostMalloc
_hipHostMalloc.argtypes = [POINTER(c_void_p), c_size_t, c_uint]
_hipHostMalloc.restype = c_int

_hipHostFree = libamdhip.hipHostFree
_hipHostFree.argtypes = [c_void_p]
_hipHostFree.restype = c_int

_hipMemset = libamdhip.hipMemset
_hipMemset.argtypes = [c_void_p, c_int, c_size_t]
_hipMemset.restype = c_int

_hipMemcpyHtoD = libamdhip.hipMemcpyHtoD
_hipMemcpyHtoD.argtypes = [c_void_p, c_void_p, c_size_t]
_hipMemcpyHtoD.restype = c_int

_hipMemcpyDtoH = libamdhip.hipMemcpyDtoH
_hipMemcpyDtoH.argtypes = [c_void_p, c_void_p, c_size_t]
_hipMemcpyDtoH.restype = c_int

#     typedef enum hipMemcpyKind {
#     hipMemcpyHostToHost,
#     hipMemcpyHostToDevice,
#     hipMemcpyDeviceToHost,
#     hipMemcpyDeviceToDevice,
#     hipMemcpyDefault
#     } hipMemcpyKind;
_hipMemcpyDefault = 4
_hipMemcpy = libamdhip.hipMemcpy
_hipMemcpy.argtypes = [c_void_p, c_void_p, c_size_t, c_int]
_hipMemcpy.restype = c_int

_hipGetDeviceCount = libamdhip.hipGetDeviceCount
_hipGetDeviceCount.argtypes = [POINTER(c_int)]
_hipGetDeviceCount.restype = c_int

_hipGetDeviceProperties = libamdhip.hipGetDeviceProperties
_hipGetDeviceProperties.argtypes = [POINTER(hipDeviceProp_t), c_int]
_hipGetDeviceProperties.restype = c_int

_hipSetDevice = libamdhip.hipSetDevice
_hipSetDevice.argtypes = [c_int]
_hipSetDevice.restype = c_int

_hipGetDevice = libamdhip.hipGetDevice
_hipGetDevice.argtypes = [POINTER(c_int)]
_hipGetDevice.restype = c_int

_hipDeviceSynchronize = libamdhip.hipDeviceSynchronize
_hipDeviceSynchronize.restype = c_int


def error_msg(func_name, error):
    err_str = _hipGetErrorString(error)
    return "ERROR: {} reported {}".format(func_name, err_str.decode())


### ========================== MEMORY ============================ ###


def hip_memory_info() -> Tuple[int, int]:
    """
    Get the free and total memory on the current device in bytes.

    Returns
    -------
    (free, total): Tuple[int, int]
        The free and total memory on the current device in bytes.
    """
    free = c_size_t()
    total = c_size_t()
    error = _hipMemGetInfo(byref(free), byref(total))
    return error, free.value, total.value


def hip_malloc(num_bytes: int) -> int:
    """
    Allocate memory on the current device.

    Parameters
    ----------
    num_bytes: int
        The number of bytes to allocate.

    Returns
    -------
    addr: int
        The address of the allocated memory.
    """
    ret = c_void_p()
    error = _hipMalloc(byref(ret), num_bytes)
    return error, ret.value


def hip_free(addr: int) -> None:
    """
    Free memory on the current hip device.

    Parameters
    ----------
    addr: int
        The address of the memory to free. This must be the address of memory allocated with :func:`malloc` or
        :func:`malloc_async`.
    """
    ptr = c_void_p(addr)
    error = _hipFree(ptr)
    return error


def hip_malloc_host(num_bytes: int) -> int:
    """
    Allocate pinned host memory.

    Parameters
    ----------
    num_bytes: int
        The number of bytes to allocate.

    Returns
    -------
    addr: int
        The address of the allocated memory.
    """
    ret = c_void_p()
    error = _hipHostMalloc(byref(ret), num_bytes, 0)  # flag == 0: hipHostMallocDefault
    return error, ret.value


def hip_free_host(addr: int) -> None:
    """
    Free pinned host memory.

    Parameters
    ----------
    addr: int
        The address of the memory to free. This must be the address of memory allocated with :func:`malloc_host`.
    """
    ptr = c_void_p(addr)
    error = _hipHostFree(ptr)
    return error


def hip_memset(addr: int, value: int, num_bytes: int) -> None:
    """
    Set the gpu memory to a given value.

    Parameters
    ----------
    addr: int
        The start address of the memory region to set.

    value: int
        The byte value to set the memory region to.

    num_bytes: int
        The number of bytes to set.
    """
    ptr = c_void_p(addr)
    error = _hipMemset(ptr, value, num_bytes)
    return error


def hip_memcpy_host_to_device(dst: int, src: int, num_bytes: int) -> None:
    """
    Copy data from host memory into device memory.

    Parameters
    ----------
    dst: int
        The destination (device) address.

    src: int
        The source (host) address.

    num_bytes: int
        The number of bytes to copy.
    """
    dst_ptr, src_ptr = c_void_p(dst), c_void_p(src)
    error = _hipMemcpyHtoD(dst_ptr, src_ptr, num_bytes)

    return error


def hip_memcpy_device_to_host(dst: int, src: int, num_bytes: int) -> None:
    """
    Copy data from host memory into device memory.

    Parameters
    ----------
    dst: int
        The destination (host) address.

    src: int
        The source (device) address.

    num_bytes: int
        The number of bytes to copy.
    """
    dst_ptr, src_ptr = c_void_p(dst), c_void_p(src)
    error = _hipMemcpyDtoH(dst_ptr, src_ptr, num_bytes)

    return error


def hip_memcpy(dst: int, src: int, num_bytes: int) -> None:
    """
    Copy data from src to dst. Uses hipMemcpyDefault to automatically determine the type of copy (DtoH, HtoD, DtoD, or
    HtoH).

    Parameters
    ----------
    dst: int
        The destination (host) address.

    src: int
        The source (device) address.

    num_bytes: int
        The number of bytes to copy.
    """
    dst_ptr, src_ptr = c_void_p(dst), c_void_p(src)
    error = _hipMemcpy(dst_ptr, src_ptr, num_bytes, _hipMemcpyDefault)
    return error


### =============================================================== ###

### ========================== DEVICES ============================ ###


def hip_device_count() -> int:
    """
    Get the number of available HIP devices.

    Returns
    -------
    count: int
        The number of available HIP devices.
    """
    ret = c_int()
    error = _hipGetDeviceCount(byref(ret))
    return error, ret.value


def hip_device_properties(device_id: int = 0) -> hipDeviceProp_t:
    """
    Get the properties of a HIP device.

    Parameters
    ----------
    device_id: int
        The ID of the device.

    Returns
    -------
    prop: hipDeviceProp_t
        The properties of the device.
    """
    prop = hipDeviceProp_t()
    error = _hipGetDeviceProperties(byref(prop), device_id)
    return error, prop


def hip_set_device(device_id: int):
    """
    Set the current HIP device.

    Parameters
    ----------
    device_id: int
        The ID of the HIP device.
    """
    error = _hipSetDevice(device_id)
    return error


def hip_current_device() -> int:
    """
    Get the current HIP device.

    Returns
    -------
    device_id: int
        The ID of the HIP device.
    """
    device_id = c_int()
    error = _hipGetDevice(byref(device_id))

    return error, device_id.value


def hip_device_synchronize():
    """
    Synchronize the host thread with the device.

    This function blocks until the device has completed all preceding requested tasks.
    """
    error = _hipDeviceSynchronize()
    return error
