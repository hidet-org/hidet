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
#include <string>
#include <vector>
#include <unordered_map>
#include <hidet/runtime/callbacks.h>
#include <hidet/runtime/logging.h>

struct CallbackRegistryPool {
    /* maps callback function name to the index of the function */
    std::unordered_map<std::string, int> name2id;
    std::unordered_map<int, std::string> id2name;
    /* array contains the function pointer of callback function */
    std::vector<void*> id2ptr;

    CallbackRegistryPool() {
        name2id["allocate_cuda_storage"] = 0;
        name2id["free_cuda_storage"] = 1;
        name2id["allocate_cpu_storage"] = 2;
        name2id["free_cpu_storage"] = 3;
        name2id["cuda_memset"] = 4;

        for (auto& kv : name2id) {
            id2name[kv.second] = kv.first;
        }
    }

    static CallbackRegistryPool* global() {
        static CallbackRegistryPool instance;
        return &instance;
    }
};

template<int id, typename FuncType>
static FuncType* get_callback_ptr() {
    auto pool = CallbackRegistryPool::global();
    assert(id < pool->id2name.size());
    if(id >= pool->id2ptr.size() || pool->id2ptr[id] == nullptr) {
        throw HidetException(__FILE__, __LINE__, "Callback function " + pool->id2name[id] + " has not been registered.");
    }
    void* ptr = pool->id2ptr[id];
    typedef FuncType* FuncPointerType;
    return FuncPointerType(ptr);
}

DLL void register_callback(const char* name, void *func_ptr) {
    try {
        auto pool = CallbackRegistryPool::global();
        if (pool->name2id.count(name) == 0) {
            throw HidetException(__FILE__, __LINE__, "Function " + std::string(name) + " is not a callback function.");
        }
        int id = pool->name2id[name];
        if(id >= pool->id2ptr.size()) {
            pool->id2ptr.resize(id + 1);
        }
        pool->id2ptr[id] = func_ptr;
    } catch (HidetException &e) {
        hidet_set_last_error(e.what());
    }
}

DLL uint64_t allocate_cuda_storage(uint64_t nbytes) {
    try {
        return get_callback_ptr<0, decltype(allocate_cuda_storage)>()(nbytes);
    } catch (HidetException &e) {
        hidet_set_last_error(e.what());
        return 0;
    }
}

DLL void free_cuda_storage(uint64_t ptr) {
    try {
        get_callback_ptr<1, decltype(free_cuda_storage)>()(ptr);
    } catch (HidetException &e) {
        hidet_set_last_error(e.what());
    }
}

DLL uint64_t allocate_cpu_storage(uint64_t nbytes) {
    try {
        return get_callback_ptr<2, decltype(allocate_cpu_storage)>()(nbytes);
    } catch (HidetException &e) {
        hidet_set_last_error(e.what());
        return 0;
    }
}

DLL void free_cpu_storage(uint64_t ptr) {
    try {
        get_callback_ptr<3, decltype(free_cpu_storage)>()(ptr);
    } catch (HidetException &e) {
        hidet_set_last_error(e.what());
    }
}

DLL void cuda_memset(uint64_t ptr, int value, uint64_t nbytes) {
    try {
        get_callback_ptr<4, decltype(cuda_memset)>()(ptr, value, nbytes);
    } catch (HidetException &e) {
        hidet_set_last_error(e.what());
    }
}
