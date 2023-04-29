// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include <hidet/runtime/callbacks.h>
#include <hidet/runtime/cuda/context.h>
#include <hidet/runtime/logging.h>

CudaContext *CudaContext::global() {
    static CudaContext instance;
    return &instance;
}

static void reserve_cuda_workspace(Workspace &workspace, size_t nbytes) {
    if(nbytes > workspace.allocated_nbytes) {
        if(workspace.base) {
            free_cuda_storage(reinterpret_cast<uint64_t>(workspace.base));
        }
        workspace.base = reinterpret_cast<void*>(allocate_cuda_storage(nbytes));
        if(workspace.base == nullptr) {
            throw HidetException(__FILE__, __LINE__, "allocate workspace failed.");
        }

        cuda_memset(reinterpret_cast<uint64_t>(workspace.base), 0, nbytes);
    }
}

DLL void set_cuda_stream(void* stream) {
    CudaContext::global()->stream = stream;
}

DLL void* get_cuda_stream() {
    return CudaContext::global()->stream;
}

DLL void* request_cuda_workspace(size_t nbytes, bool require_clean) {
    try {
        auto ctx = CudaContext::global();
        if(require_clean) {
            reserve_cuda_workspace(ctx->clean_workspace, nbytes);
            return ctx->clean_workspace.base;
        } else {
            reserve_cuda_workspace(ctx->dirty_workspace, nbytes);
            return ctx->dirty_workspace.base;
        }
    } catch (HidetException &e) {
        hidet_set_last_error(e.what());
        return nullptr;
    }
}
