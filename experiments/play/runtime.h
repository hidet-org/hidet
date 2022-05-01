#include <cstdio>

#define DLL extern "C" __attribute__ ((visibility ("default"))) 


extern int global_var;
DLL void func();
