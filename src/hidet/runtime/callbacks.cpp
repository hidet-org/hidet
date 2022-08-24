#include <string>
#include <vector>
#include <unordered_map>
#include <hidet/runtime/callbacks.h>
#include <hidet/runtime/logging.h>

struct CallbackRegistryPool {
    std::unordered_map<std::string, int> name2id;
    std::vector<void*> id2ptr;

    CallbackRegistryPool() {
        name2id["allocate_cuda_storage"] = 0;
        name2id["free_cuda_storage"] = 1;
    }

    static CallbackRegistryPool* global() {
        static thread_local CallbackRegistryPool instance;
        return &instance;
    }
};

template<int id, typename FuncType>
static FuncType* get_callback_ptr() {
    auto pool = CallbackRegistryPool::global();
    if(pool->id2ptr.size() >= id) {
        throw HidetException(__FILE__, __LINE__, "Callback function registry pool has not been populated.");
    }
    void* ptr = pool->id2ptr[id];
    if (ptr == nullptr) {
        throw HidetException(__FILE__, __LINE__, "Callback with id " + std::to_string(id) + " has been called before registration.");
    }
    typedef FuncType& FuncPointerType;
    return FuncPointerType(ptr);
}

DLL void register_callback(const char* name, void *func_ptr) {
    API_BEGIN()
    auto pool = CallbackRegistryPool::global();
    if (pool->name2id.count(name) == 0) {
        throw HidetException(__FILE__, __LINE__, "Function " + std::string(name) + " is not a callback function.");
    }
    int id = pool->name2id[name];
    if(id >= pool->id2ptr.size()) {
        pool->id2ptr.resize(id + 1);
    }
    pool->id2ptr[id] = func_ptr;
    API_END()
}

DLL uint64_t allocate_cuda_storage(uint64_t nbytes) {
    API_BEGIN()
    return get_callback_ptr<0, typeof(allocate_cuda_storage)>()(nbytes);
    API_END(0)
}

DLL void free_cuda_storage(uint64_t ptr) {
    API_BEGIN()
    return get_callback_ptr<0, typeof(free_cuda_storage)>()(ptr);
    API_END()
}

