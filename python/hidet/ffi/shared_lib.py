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
from typing import Dict

# retrieve the dlclose function
_dlclose = ctypes.CDLL(None).dlclose  # None indicates the main program
_dlclose.argtypes = [ctypes.c_void_p]
_dlclose.rettype = ctypes.c_int


class SharedLibrary:
    """
    Manage the loaded dynamic libraries.

    Why we need this module?
    ------------------------

    The ctypes.CDLL class does not provide a way to unload the loaded library. When a library is loaded, it will never
    be unloaded until the program exits. However, when we tune an operator, we need to generate hundreds of kernels, and
    each kernel will be compiled into a shared library. If we do not unload the shared library, we would load tens of
    thousands of shared libraries, which will trigger memory error like:
      "cannot apply additional memory protection after relocation"
    (I also see other error messages).
    To solve this problem, we need to unload the shared library after we use it. Thus, whenever we need to load a shared
    library, we should use this module instead of the ctypes.CDLL class. The SharedLibrary class will keep track of the
    loaded libraries, and unload them when no one references them.

    The current implementation only supports *nix systems. Will add support for Windows when we plan to support Windows
    in the project-level.

    Usage
    -----

    >>> lib = SharedLibrary('./libhidet.so')
    >>> func = lib['func_name']
    >>> del func
    >>> del lib
    >>> # the 'libhidet.so' will be unloaded via dlclose after the last reference to it is deleted.
    """

    loaded_cdll_libraries: Dict[str, ctypes.CDLL] = {}
    reference_count: Dict[str, int] = {}

    def __init__(self, lib_path: str):
        self.lib_path: str = lib_path
        cls = SharedLibrary
        if lib_path in cls.loaded_cdll_libraries:
            self.cdll: ctypes.CDLL = cls.loaded_cdll_libraries[lib_path]
            cls.reference_count[lib_path] += 1
        else:
            cdll = ctypes.CDLL(lib_path)
            self.cdll: ctypes.CDLL = cdll
            cls.loaded_cdll_libraries[lib_path] = cdll
            cls.reference_count[lib_path] = 1

    def __getitem__(self, item):
        """
        Get the function from the loaded library.

        Parameters
        ----------
        item: str
            The name of the function.

        Returns
        -------
        func: ctypes.CFUNCTYPE
            The loaded function.
        """
        ret = self.cdll[item]
        ret._lib = self
        return ret

    def __getattr__(self, item):
        return self[item]

    def __del__(self):
        self.reference_count[self.lib_path] -= 1
        if self.reference_count[self.lib_path] == 0:
            del self.loaded_cdll_libraries[self.lib_path]
            del self.reference_count[self.lib_path]
            _dlclose(self.cdll._handle)
            del self.cdll
