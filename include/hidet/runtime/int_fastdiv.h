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

#include <hidet/runtime/common.h>

#ifdef __CUDA_ARCH__
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif

HOST_DEVICE void calculate_magic_numbers(int d, int &m, int &s, int &as);
