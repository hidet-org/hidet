import ctypes


class LibCAPI:
    """
    A wrapper class for C functions in libc.
    This module is needed when CUDA is not available so we switch to 
    using the C runtime library to do memory management.

    Currently only support *nix systems.
    """
    # init by loading libc into a ctypes.CDLL object
    def __init__(self):
        self.libc = ctypes.CDLL("libc.so.6")

    # define a function that calls the C function malloc
    def malloc(self, size: int) -> ctypes.c_void_p:
        self.libc.malloc.argtypes = [ctypes.c_size_t]
        self.libc.malloc.restype = ctypes.c_void_p
        return self.libc.malloc(size)
    
    # define a function that calls the C function free
    def free(self, ptr: int) -> None:
        self.libc.free.argtypes = [ctypes.c_void_p]
        self.libc.free.restype = None
        return self.libc.free(ptr)

_LIBCAPI: Optional[LibCAPI] = None

def load_libc() -> None:
    global _LIBCAPI
    if _LIBCAPI:
        return
    try:
        _LIBCAPI = LibCAPI()
    except OSError as e:
        print("Failed to load libc.so.6")
        raise e

load_libc()


# expose libc_malloc and libc_free to storage
def libc_malloc(size: int) -> int:
    return int(_LIBCAPI.malloc(size))

def libc_free(ptr: int) -> None:
    _LIBCAPI.free(ctypes.c_void_p(ptr))