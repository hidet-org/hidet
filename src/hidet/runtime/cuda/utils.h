#include <dlfcn.h>
#include <hidet/runtime/logging.h>

template<typename T>
inline T get_symbol(void *lib, const char *name) {
    T ret = (T)dlsym(lib, name);
    if (ret == nullptr) {
        LOG(FATAL) << "Failed to load symbol: " << std::endl << "  " << dlerror();
    }
    return ret;
}
