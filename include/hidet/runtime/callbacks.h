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
#include <cstdint>
#include <hidet/runtime/common.h>

DLL void register_callback(const char* name, void *func_ptr);

DLL uint64_t allocate_cuda_storage(uint64_t nbytes);

DLL void free_cuda_storage(uint64_t ptr);

DLL uint64_t allocate_cpu_storage(uint64_t nbytes);

DLL void free_cpu_storage(uint64_t ptr);

DLL void cuda_memset(uint64_t ptr, int value, uint64_t nbytes);