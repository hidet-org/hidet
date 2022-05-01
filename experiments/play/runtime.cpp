#include "runtime.h"



int global_var;
DLL void func() {
    printf("runtime func %d\n", global_var++);
}
