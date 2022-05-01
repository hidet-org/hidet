#include "runtime.h"


DLL void main_func() {
    printf("main func %d\n", global_var);
    global_var += 1;
    // func();
}
