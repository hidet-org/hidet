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
import ctypes
from typing import Optional


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

    def malloc(self, size: int) -> int:
        """
        Allocate cpu memory.

        Parameters
        ----------
        size: int
            The number of bytes to allocate.

        Returns
        -------
        ret: int
            The pointer to the allocated memory.
        """
        self.libc.malloc.argtypes = [ctypes.c_size_t]
        self.libc.malloc.restype = ctypes.c_void_p
        return int(self.libc.malloc(size))

    def free(self, addr: int) -> None:
        """
        Free cpu memory.

        Parameters
        ----------
        addr: int
            The pointer to the memory to be freed. The pointer must be returned by malloc.
        """
        self.libc.free.argtypes = [ctypes.c_void_p]
        self.libc.free.restype = None
        return self.libc.free(addr)


_LIBCAPI: Optional[LibCAPI] = None


def lazy_load_libc() -> None:
    global _LIBCAPI
    if _LIBCAPI:
        return
    try:
        _LIBCAPI = LibCAPI()
    except OSError as e:
        print("Failed to load C runtime library.")
        raise e


# expose libc_malloc and libc_free to storage
def malloc(size: int) -> int:
    lazy_load_libc()
    return _LIBCAPI.malloc(size)


def free(ptr: int) -> None:
    lazy_load_libc()
    _LIBCAPI.free(ctypes.c_void_p(ptr))
