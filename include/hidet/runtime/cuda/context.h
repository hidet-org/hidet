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
#pragma once
#include <hidet/runtime/common.h>
#include <hidet/runtime/callbacks.h>
#include <hidet/runtime/context.h>
// #include <cuda_runtime.h>

#include <vector>

struct CudaContext: BaseContext {
    /* The cuda stream the kernels will be launched on. */
    void* stream = nullptr;

    /* NCCL Comunicators*/
    std::vector<void *> nccl_comms;

    /**
     * Get the instance of cuda context.
     */
    static CudaContext* global();
};

/**
 * Set the cuda stream of cuda context.
 */
DLL void set_cuda_stream(void* stream);

/**
 * Get the cuda stream of cuda context.
 */
DLL void* get_cuda_stream();

/**
 * Request a workspace.
 */
DLL void* request_cuda_workspace(size_t nbytes, bool require_clean);

/**
 * Add a NCCL communicator to the context.
 */
DLL void add_nccl_comm(void* comm);

/**
 * Get the NCCL communicator by the index
 */
DLL void* get_nccl_comm(int idx);