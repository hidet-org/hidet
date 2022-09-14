#pragma once

#include <hidet/runtime/context.h>

struct CpuContext: BaseContext {
    static CpuContext* global();
};


/**
 * Request a workspace.
 */
DLL void* request_cpu_workspace(size_t nbytes, bool require_clean);

