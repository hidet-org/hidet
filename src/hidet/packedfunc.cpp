#include <cstdio>
#include <hidet/packedfunc.h>
#include <hidet/runtime.h>

extern "C" {

DLL void CallPackedFunc(PackedFunc func, void** args) {
    auto f = PackedFunc_t(func.func_pointer);
    f(func.num_args, func.arg_types, args);
}

}

