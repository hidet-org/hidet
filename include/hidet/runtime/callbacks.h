#include <cstdint>
#include <hidet/runtime/common.h>

DLL void register_callback(const char* name, void *func_ptr);

DLL uint64_t allocate_cuda_storage(uint64_t nbytes);

DLL void free_cuda_storage(uint64_t ptr);

DLL uint64_t allocate_cpu_storage(uint64_t nbytes);

DLL void free_cpu_storage(uint64_t ptr);

DLL void cuda_memset(uint64_t ptr, int value, uint64_t nbytes);