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
#include <hidet/runtime/callbacks.h>
#include <hidet/runtime/common.h>
#include <hidet/runtime/context.h>

struct HipContext: BaseContext {
    /* The hip stream the kernels will be launched on. */
    void *stream = nullptr;

    /**
     * Get the instance of hip context.
     */
    static HipContext *global();
};

/**
 * Set the hip stream of hip context.
 */
DLL void set_hip_stream(void *stream);

/**
 * Get the hip stream of hip context.
 */
DLL void *get_hip_stream();

/**
 * Request a workspace.
 */
DLL void *request_hip_workspace(size_t nbytes, bool require_clean);
