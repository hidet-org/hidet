#pragma once
#include <iostream>
#include <cassert>


//#if defined(assert) && !defined(__CUDA_ARCH__)
//// redefine assert to give more information where the assertion failed in host code.
//#undef assert
//#define assert(x) if(!(x)){                                        \
//        std::cerr << __FILE__ << ":" << __LINE__ << ": "           \
//        << #x << "failed" << std::endl;                            \
//        exit(-1);                                                  \
//}
//#endif

#ifndef DLL
#define DLL extern "C" __attribute__((visibility("default")))
#endif

