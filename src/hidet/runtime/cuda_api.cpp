#include <cstdint>
#include <ctime>
#include <hidet/runtime.h>
#include <cuda_runtime.h>
#include <curand.h>

struct CurandContext {
    curandGenerator_t generator;
    CurandContext() {
        unsigned long long seed = time(nullptr) ^ clock();
        CURAND_CALL(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT));
        CURAND_CALL(curandSetPseudoRandomGeneratorSeed(generator, seed));
    }

    static CurandContext* global() {
        static CurandContext ctx;
        return &ctx;
    }
};


DLL uint64_t hidet_cuda_malloc_async(uint64_t bytes) {
    void *ptr;
    CUDA_CALL(cudaMallocAsync(&ptr, bytes, nullptr));
    return reinterpret_cast<uint64_t>(ptr);
}

DLL uint64_t hidet_cuda_malloc_host(uint64_t bytes) {
    void* ptr;
    CUDA_CALL(cudaMallocHost(&ptr, bytes));
    return reinterpret_cast<uint64_t>(ptr);
}

DLL void hidet_cuda_free_async(uint64_t addr) {
    CUDA_CALL(cudaFreeAsync(reinterpret_cast<void*>(addr), nullptr));
}

DLL void hidet_cuda_free_host(uint64_t addr) {
    CUDA_CALL(cudaFreeHost(reinterpret_cast<void*>(addr)));
}

DLL void hidet_cuda_memset_async(uint64_t addr, uint64_t bytes, uint8_t value) {
    CUDA_CALL(cudaMemsetAsync(reinterpret_cast<void*>(addr), value, bytes, nullptr));
}

DLL void hidet_cuda_memcpy_async(uint64_t src, uint64_t dst, uint64_t bytes, uint32_t kind) {
    /*!
     * kind:
     *   cudaMemcpyHostToHost          =   0,
     *   cudaMemcpyHostToDevice        =   1,
     *   cudaMemcpyDeviceToHost        =   2,
     *   cudaMemcpyDeviceToDevice      =   3,
    */
    CUDA_CALL(cudaMemcpyAsync(reinterpret_cast<void*>(dst), reinterpret_cast<void*>(src), bytes, cudaMemcpyKind(kind), nullptr));
}

DLL void hidet_cuda_device_synchronization() {
    CUDA_CALL(cudaDeviceSynchronize());
}

DLL void hidet_curand_generate_uniform(uint64_t addr, uint64_t size) {
    CURAND_CALL(curandGenerateUniform(CurandContext::global()->generator, reinterpret_cast<float*>(addr), size));
}

DLL void hidet_curand_generate_normal(uint64_t addr, uint64_t size, float mean, float stddev) {
    // This function only support to generate even number of random numbers. We work around this limitation by up round to a multiple of 2.
    // this usually will not trigger error because the memory allocation on cuda is usually 256 bytes aligned.
    if(size & 1) {
        size += 1;
    }
    CURAND_CALL(curandGenerateNormal(CurandContext::global()->generator, reinterpret_cast<float*>(addr), size, mean, stddev));
}

DLL void hidet_cuda_mem_pool_trim_to(uint64_t min_bytes_to_keep) {
    cudaMemPool_t pool;
    CUDA_CALL(cudaDeviceGetDefaultMemPool(&pool, 0));
    CUDA_CALL(cudaMemPoolTrimTo(pool, min_bytes_to_keep));
}