import nvtx

nvtx_annotate = nvtx.annotate


class CudaProfileContext:
    def __enter__(self):
        from hidet.ffi import cuda

        cuda.start_profiler()

    def __exit__(self, exc_type, exc_val, exc_tb):
        from hidet.ffi import cuda

        cuda.stop_profiler()


def enable_cuda_profile():
    return CudaProfileContext()
