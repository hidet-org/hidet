// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#pragma once

#ifndef DLL
#define DLL extern "C" __attribute__((visibility("default")))
#endif

enum ArgType {
    INT32 = 1,
    FLOAT32 = 2,
    POINTER = 3,
};

typedef void (*PackedFunc_t)(int num_args, int *arg_types, void** args);

struct PackedFunc {
    int num_args;
    int* arg_types;
    void** func_pointer;
};

#define INT_ARG(p) (*(int*)(p))
#define FLOAT_ARG(p) (*(float*)(p))


