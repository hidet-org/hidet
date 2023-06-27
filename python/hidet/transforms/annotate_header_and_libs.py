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
import hidet.ir
from hidet.ir.module import IRModule
from hidet.ir.stmt import BlackBoxStmt
from hidet.transforms import Pass


def _use_distributed(func) -> bool:
    black_stmts = hidet.ir.tools.collect(func.body, [BlackBoxStmt])
    return any(stmt.template_string.startswith('nccl') for stmt in black_stmts)


class AnnotateHeaderAndLibsPass(Pass):
    def process_module(self, ir_module: IRModule) -> IRModule:
        use_dist = any(_use_distributed(func) for func in ir_module.functions.values())
        if not use_dist:
            return ir_module

        from hidet.cuda.nccl.libinfo import get_nccl_include_dirs, get_nccl_library_search_dirs
        from hidet.cuda.nccl import nccl_available, nccl_library_filename

        if not nccl_available():
            raise RuntimeError("NCCL is not available")

        new_module = ir_module.copy()
        new_module.include_dirs.extend(get_nccl_include_dirs())
        new_module.linking_dirs.extend(get_nccl_library_search_dirs())
        new_module.include_headers.append(["nccl.h"])
        new_module.linking_libs.append(":" + nccl_library_filename())
        return new_module


def annotate_header_and_libs_pass():
    return AnnotateHeaderAndLibsPass()
