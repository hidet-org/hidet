#include <cstdint>
#include <hidet/common.h>

DLL void register_callback(const char* name, void *func_ptr);

DLL uint64_t allocate_cuda_storage(uint64_t nbytes);

DLL void free_cuda_storage(uint64_t ptr);

