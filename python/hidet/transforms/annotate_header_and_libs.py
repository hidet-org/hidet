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
from typing import List
from hidet.ir.tools import collect
from hidet.ir.module import IRModule
from hidet.ir.expr import Call
from hidet.ir.stmt import BlackBoxStmt
from hidet.transforms import Pass


class Annotator:
    def __init__(self):
        self.include_dirs: List[str] = []
        self.linking_dirs: List[str] = []
        self.include_headers: List[str] = []
        self.linking_libs: List[str] = []

    def predicate(self, ir_module: IRModule) -> bool:
        raise NotImplementedError()

    def apply(self):
        raise NotImplementedError()


class AnnotateNCCL(Annotator):
    """
    Annotate the header and libraries for NCCL.

    Headers: nccl.h
    Libraries: libnccl.so
    """

    def predicate(self, ir_module: IRModule) -> bool:
        for func in ir_module.functions.values():
            black_stmts = collect(func.body, [BlackBoxStmt])
            if any(stmt.template_string.startswith('nccl') for stmt in black_stmts):
                return True
        return False

    def apply(self):
        from hidet.cuda.nccl.libinfo import get_nccl_include_dirs, get_nccl_library_search_dirs
        from hidet.cuda.nccl import nccl_available, nccl_library_filename

        if not nccl_available():
            raise RuntimeError("NCCL is not available")

        self.include_dirs.extend(get_nccl_include_dirs())
        self.linking_dirs.extend(get_nccl_library_search_dirs())
        self.include_headers.append("nccl.h")
        self.linking_libs.append(":" + nccl_library_filename())


class AnnotateCUBLAS(Annotator):
    def predicate(self, ir_module: IRModule) -> bool:
        for func in ir_module.functions.values():
            calls: List[Call] = collect(func.body, [Call])
            if any(call.func_var.name and call.func_var.name.startswith('cublas.') for call in calls):
                return True
        return False

    def apply(self):
        self.include_headers.append("hidet/runtime/cuda/cublas.h")


class AnnotateHeaderAndLibsPass(Pass):
    def process_module(self, ir_module: IRModule) -> IRModule:
        annotators = [AnnotateNCCL(), AnnotateCUBLAS()]
        include_dirs: List[str] = []
        linking_dirs: List[str] = []
        include_headers: List[str] = []
        linking_libs: List[str] = []

        for annotator in annotators:
            if annotator.predicate(ir_module):
                annotator.apply()
                include_dirs.extend(annotator.include_dirs)
                linking_dirs.extend(annotator.linking_dirs)
                include_headers.extend(annotator.include_headers)
                linking_libs.extend(annotator.linking_libs)

        new_module = ir_module.copy()
        new_module.include_dirs.extend(include_dirs)
        new_module.linking_dirs.extend(linking_dirs)
        new_module.include_headers.extend(include_headers)
        new_module.linking_libs.extend(linking_libs)
        return new_module


def annotate_header_and_libs_pass():
    return AnnotateHeaderAndLibsPass()
