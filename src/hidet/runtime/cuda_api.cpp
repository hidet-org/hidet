#include <cstdint>
#include <hidet/runtime.h>
#include <cuda_runtime.h>



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
    void *ptr = reinterpret_cast<void*>(addr);
    CUDA_CALL(cudaFreeAsync(ptr, nullptr));
}

DLL void hidet_cuda_free_host(uint64_t addr) {
    void *ptr = reinterpret_cast<void*>(addr);
    CUDA_CALL(cudaFreeHost(ptr));
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
