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
#include <hidet/runtime/torch/stream.h>

CudaContext *CudaContext::global() {
    static CudaContext instance;
    return &instance;
}

static void reserve_cuda_workspace(Workspace &workspace, size_t nbytes) {
    if (nbytes > workspace.allocated_nbytes) {
        if (workspace.base) {
            free_cuda_storage(reinterpret_cast<uint64_t>(workspace.base));
        }
        workspace.base = reinterpret_cast<void *>(allocate_cuda_storage(nbytes));
        if (workspace.base == nullptr) {
            LOG(ERROR) << "allocate workspace failed.";
        }

        cuda_memset(reinterpret_cast<uint64_t>(workspace.base), 0, nbytes);
    }
}

DLL void set_cuda_stream(void *stream) {
    CudaContext::global()->stream = stream;
}

DLL bool get_use_torch_cuda_stream() {
    return CudaContext::global()->use_torch_stream;
}

DLL void use_torch_cuda_stream(bool use) {
    CudaContext::global()->use_torch_stream = use;
}

DLL void *get_cuda_stream() {
    if (CudaContext::global()->use_torch_stream) {
        return reinterpret_cast<void *>(get_torch_stream());
    }
    return CudaContext::global()->stream;
}

DLL void *request_cuda_workspace(size_t nbytes, bool require_clean) {
    try {
        auto ctx = CudaContext::global();
        if (require_clean) {
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

DLL void set_nccl_comms(int num_comms, void **comms) {
    CudaContext::global()->num_comms = num_comms;
    CudaContext::global()->nccl_comms = comms;
}

DLL void *get_nccl_comm(int idx) {
    const int num_comms = CudaContext::global()->num_comms;
    if (idx >= num_comms) {
        LOG(ERROR) << "Index of NCCL Communicator out of boundary. (" << idx << " vs " << num_comms << ")";
    }
    return CudaContext::global()->nccl_comms[idx];
}