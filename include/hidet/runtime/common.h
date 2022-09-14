#pragma once
#include <iostream>
#include <cassert>

#ifndef DLL
#define DLL extern "C" __attribute__((visibility("default")))
#endif

