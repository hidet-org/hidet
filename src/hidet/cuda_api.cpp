#include <cstdint>
#include <ctime>
#include <hidet/common.h>
#include <hidet/cuda_utils.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <hidet/runtime/logging.h>
#include <cuda_profiler_api.h>

struct CurandContext {
    curandGenerator_t generator{};
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

DLL void hidet_cuda_mem_info(uint64_t *free, uint64_t *total) {
    API_BEGIN();
    CUDA_CALL(cudaMemGetInfo(free, total));
    API_END();
}

DLL uint64_t hidet_cuda_malloc_async(uint64_t bytes) {
    API_BEGIN();
    void *ptr;
    cudaError_t status = cudaMallocAsync(&ptr, bytes, nullptr);
    if(status == cudaErrorMemoryAllocation) {
        // out of memory
        return 0;
    }
    CUDA_CALL(status);
    return reinterpret_cast<uint64_t>(ptr);
    API_END(0);
}

DLL uint64_t hidet_cuda_malloc_host(uint64_t bytes) {
    API_BEGIN();
    void* ptr;
    CUDA_CALL(cudaMallocHost(&ptr, bytes));
    return reinterpret_cast<uint64_t>(ptr);
    API_END(0);
}

DLL void hidet_cuda_free_async(uint64_t addr) {
    API_BEGIN();
    CUDA_CALL(cudaFreeAsync(reinterpret_cast<void*>(addr), nullptr));
    API_END();
}

DLL void hidet_cuda_free_host(uint64_t addr) {
    API_BEGIN();
    CUDA_CALL(cudaFreeHost(reinterpret_cast<void*>(addr)));
//    auto status = cudaFreeHost(reinterpret_cast<void*>(addr));
//    if(status != cudaSuccess) {
//        fprintf(stderr, "Can not free host memory %p\n", reinterpret_cast<void*>(addr));
//    }
    API_END();
}

DLL void hidet_cuda_memset_async(uint64_t addr, uint64_t bytes, uint8_t value) {
    API_BEGIN();
    CUDA_CALL(cudaMemsetAsync(reinterpret_cast<void*>(addr), value, bytes, nullptr));
    API_END();
}

DLL void hidet_cuda_memcpy_async(uint64_t src, uint64_t dst, uint64_t bytes, uint32_t kind, uint64_t stream) {
    API_BEGIN();
    /*!
     * kind:
     *   cudaMemcpyHostToHost          =   0,
     *   cudaMemcpyHostToDevice        =   1,
     *   cudaMemcpyDeviceToHost        =   2,
     *   cudaMemcpyDeviceToDevice      =   3,
    */
    CUDA_CALL(cudaMemcpyAsync(reinterpret_cast<void*>(dst), reinterpret_cast<void*>(src), bytes, cudaMemcpyKind(kind), reinterpret_cast<cudaStream_t>(stream)));
    API_END();
}

DLL void hidet_cuda_device_synchronize() {
    API_BEGIN();
    CUDA_CALL(cudaDeviceSynchronize());
    API_END();
}

DLL void hidet_curand_generate_uniform(uint64_t addr, uint64_t size) {
    API_BEGIN();
    CURAND_CALL(curandGenerateUniform(CurandContext::global()->generator, reinterpret_cast<float*>(addr), size));
    API_END();
}

DLL void hidet_curand_generate_normal(uint64_t addr, uint64_t size, float mean, float stddev) {
    API_BEGIN();
    // This function only support to generate even number of random numbers. We work around this limitation by up round to a multiple of 2.
    // this usually will not trigger error because the memory allocation on cuda is 256 bytes aligned.
    if(size & 1) {
        size += 1;
    }
    CURAND_CALL(curandGenerateNormal(CurandContext::global()->generator, reinterpret_cast<float*>(addr), size, mean, stddev));
    API_END();
}

DLL void hidet_cuda_mem_pool_trim_to(uint64_t min_bytes_to_keep) {
    API_BEGIN();
    cudaMemPool_t pool;
    CUDA_CALL(cudaDeviceGetDefaultMemPool(&pool, 0));
    CUDA_CALL(cudaMemPoolTrimTo(pool, min_bytes_to_keep));
    API_END();
}

DLL uint64_t hidet_cuda_stream_create() {
    API_BEGIN();
    cudaStream_t stream;
    CUDA_CALL(cudaStreamCreate(&stream));
    return reinterpret_cast<uint64_t>(stream);
    API_END(0);
}


DLL void hidet_cuda_stream_destroy(uint64_t stream) {
    API_BEGIN();
    if(stream != 0) {
        CUDA_CALL(cudaStreamDestroy(reinterpret_cast<cudaStream_t>(stream)));
    }
    API_END();
}

DLL void hidet_cuda_stream_synchronize(uint64_t stream) {
    API_BEGIN();
    CUDA_CALL(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)));
    API_END();
}

DLL uint64_t hidet_cuda_event_create() {
    API_BEGIN();
    cudaEvent_t event;
    CUDA_CALL(cudaEventCreate(&event));
    return reinterpret_cast<uint64_t>(event);
    API_END(0);
}

DLL void hidet_cuda_event_destroy(uint64_t handle) {
    API_BEGIN();
    auto event = reinterpret_cast<cudaEvent_t>(handle);
    CUDA_CALL(cudaEventDestroy(event));
    API_END();
}

DLL float hidet_cuda_event_elapsed_time(uint64_t start, uint64_t end) {
    API_BEGIN();
    float latency;
    CUDA_CALL(cudaEventElapsedTime(&latency, reinterpret_cast<cudaEvent_t>(start), reinterpret_cast<cudaEvent_t>(end)));
    return latency;
    API_END(0.0);
}

DLL void hidet_cuda_event_record(uint64_t event_handle, uint64_t stream_handle) {
    API_BEGIN();
    CUDA_CALL(cudaEventRecord(reinterpret_cast<cudaEvent_t>(event_handle), reinterpret_cast<cudaStream_t>(stream_handle)));
    API_END();
}


DLL uint64_t hidet_cuda_graph_create() {
    API_BEGIN();
    cudaGraph_t graph;
    CUDA_CALL(cudaGraphCreate(&graph, 0));
    return reinterpret_cast<uint64_t>(graph);
    API_END(0);
}

DLL void hidet_cuda_graph_destroy(uint64_t handle) {
    API_BEGIN();
    CUDA_CALL(cudaGraphDestroy(reinterpret_cast<cudaGraph_t>(handle)));
    API_END();
}

DLL void hidet_cuda_stream_begin_capture(uint64_t stream_handle) {
    API_BEGIN();
    CUDA_CALL(cudaStreamBeginCapture(reinterpret_cast<cudaStream_t>(stream_handle), cudaStreamCaptureModeThreadLocal));
    API_END();
}

DLL uint64_t hidet_cuda_stream_end_capture(uint64_t stream_handle) {
    API_BEGIN();
    cudaGraph_t graph;
    CUDA_CALL(cudaStreamEndCapture(reinterpret_cast<cudaStream_t>(stream_handle), &graph));
    return reinterpret_cast<uint64_t>(graph);
    API_END(0);
}

DLL uint64_t hidet_cuda_graph_instantiate(uint64_t graph_handle) {
    API_BEGIN();
    auto graph = reinterpret_cast<cudaGraph_t>(graph_handle);
    cudaGraphExec_t graph_exec;
    CUDA_CALL(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
    return reinterpret_cast<uint64_t>(graph_exec);
    API_END(0);
}

DLL void hidet_cuda_graph_exec_launch(uint64_t graph_exec_handle, uint64_t stream_handle) {
    API_BEGIN();
    CUDA_CALL(cudaGraphLaunch(reinterpret_cast<cudaGraphExec_t>(graph_exec_handle),
                              reinterpret_cast<cudaStream_t>(stream_handle)));
    API_END();
}

DLL void hidet_cuda_graph_exec_destroy(uint64_t graph_exec_handle) {
    API_BEGIN();
    CUDA_CALL(cudaGraphExecDestroy(reinterpret_cast<cudaGraphExec_t>(graph_exec_handle)));
    API_END();
}

DLL void hidet_cuda_profiler_start() {
    API_BEGIN();
    CUDA_CALL(cudaProfilerStart());
    API_END();
}

DLL void hidet_cuda_profiler_stop() {
    API_BEGIN();
    CUDA_CALL(cudaProfilerStop());
    API_END();
}

DLL uint64_t hidet_cuda_get_device_property(uint64_t device_id, const char *property_name) {
    API_BEGIN();
    static bool queried = false;
    static cudaDeviceProp prop{};
    if(!queried) {
        CUDA_CALL(cudaGetDeviceProperties(&prop, device_id));
    }

    std::string name(property_name);
    if(name == "multiProcessorCount") {
        return prop.multiProcessorCount;
    } else if(name == "major") {
        return prop.major;
    } else if(name == "minor") {
        return prop.minor;
    } else {
        std::cout << "Can not recognize property name: " << name << std::endl;
        return 0;
    }
    API_END(0);
}
