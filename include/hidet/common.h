#pragma once
#include <iostream>
#ifdef assert
#undef assert
#endif
#define assert(x) if(!(x)){                                        \
        std::cerr << __FILE__ << ":" << __LINE__ << ": "           \
        << #x << "failed" << std::endl;                            \
        exit(-1);                                                  \
}

#ifndef DLL
#define DLL extern "C" __attribute__((visibility("default")))
#endif

