#include <cassert>
#include <cstdio>

#include <hidet/packedfunc.h>

extern "C" {

DLL void CallPackedFunc(PackedFunc func, void** args) {
    auto f = PackedFunc_t(func.func_pointer);
//    for(int i = 0; i < func.num_args; i++) {
//        switch (func.arg_types[i]) {
//            case INT32:
//                fprintf(stderr, "Arg %d: %d\n", i, *(int*)args[i]);
//                break;
//            case FLOAT32:
//                fprintf(stderr, "Arg %d: %f\n", i, *(float*)args[i]);
//                break;
//            case POINTER:
//                fprintf(stderr, "Arg %d: %p\n", i, args[i]);
//                break;
//        }
//    }
    f(func.num_args, func.arg_types, args);
}

DLL void hello_world(int num_args, int *arg_types, void** args) {
    assert(num_args == 1);
    int type_code = arg_types[0];
    assert(type_code == INT32);
    int* arg = static_cast<int *>(args[0]);
    printf("hello, world!\n%d\n", *arg);
}

}

