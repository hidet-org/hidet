#pragma once

#include <hidet/runtime/common.h>
#include <hidet/runtime/callbacks.h>

struct Workspace {
    void* base;
    size_t allocated_nbytes;
    Workspace() {
        base = nullptr;
        allocated_nbytes = 0;
    }
};

struct BaseContext {
    /* The clean workspace. The buffer only contains zero values. */
    Workspace clean_workspace;
    /* The dirty workspace. The buffer contains arbitrary values. */
    Workspace dirty_workspace;
};
