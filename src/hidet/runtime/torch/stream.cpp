#include <dlfcn.h>
#include "../cuda/utils.h"

typedef void *(*hidetGetCurrentCUDAStream_t)();
static hidetGetCurrentCUDAStream_t hidetGetCurrentTorchCUDAStream = nullptr;

static std::string library_path;
static void *libhidet_torch_wrapper = nullptr;
// load torch runtime APIs
static inline void lazy_load_torch_runtime() {
    if (libhidet_torch_wrapper == nullptr) {
        const char *libpath;
        if (library_path.empty()) {
            libpath = "libhidet_torch_wrapper.so";
        } else {
            libpath = library_path.c_str();
        }
        libhidet_torch_wrapper = dlopen(libpath, RTLD_LAZY);

        if (libhidet_torch_wrapper == nullptr) {
            LOG(FATAL) << "Failed to load libhidet_torch_wrapper.so: " << dlerror();
        }

        hidetGetCurrentTorchCUDAStream =
            get_symbol<hidetGetCurrentCUDAStream_t>(libhidet_torch_wrapper, "hidet_get_current_torch_cuda_stream");
    }
}

DLL void *hidet_get_current_torch_stream() {
    lazy_load_torch_runtime();
    return hidetGetCurrentTorchCUDAStream();
}

// Hidet exported APIs
DLL void hidet_torch_set_library_path(const char *path) {
    library_path = path;
}