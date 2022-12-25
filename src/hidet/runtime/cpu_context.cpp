#include <hidet/runtime/logging.h>
#include <hidet/runtime/cpu_context.h>
#include <cstring>


CpuContext* CpuContext::global() {
    static CpuContext instance;
    return &instance;
}

static void reserve_cpu_workspace(Workspace &workspace, size_t nbytes) {
    if(nbytes > workspace.allocated_nbytes) {
        if(workspace.base) {
            free_cuda_storage(reinterpret_cast<uint64_t>(workspace.base));
        }
        workspace.base = reinterpret_cast<void*>(allocate_cpu_storage(nbytes));
        if(workspace.base == nullptr) {
            throw HidetException(__FILE__, __LINE__, "allocate workspace failed.");
        }
        memset(workspace.base, 0, nbytes);
    }
}

DLL void* request_cpu_workspace(size_t nbytes, bool require_clean) {
    try {
        auto ctx = CpuContext::global();
        if(require_clean) {
            reserve_cpu_workspace(ctx->clean_workspace, nbytes);
            return ctx->clean_workspace.base;
        } else {
            reserve_cpu_workspace(ctx->dirty_workspace, nbytes);
            return ctx->dirty_workspace.base;
        }
    } catch (HidetException &e) {
        hidet_set_last_error(e.what());
        return nullptr;
    }
}
