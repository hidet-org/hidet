from __future__ import annotations
from typing import List
from hidet.ffi import cuda
from hidet.ffi import runtime_api


class CudaStream:
    stack: List[CudaStream] = []

    def __init__(self):
        self.handle = cuda.create_stream()

    def __enter__(self):
        CudaStream.stack.append(self)
        runtime_api.set_current_stream(self.handle)

    def __exit__(self, exc_type, exc_val, exc_tb):
        CudaStream.stack.pop()
        if len(CudaStream.stack) == 0:
            # set to default stream
            runtime_api.set_current_stream(0)
        else:
            cur_stream = CudaStream.stack[-1]
            runtime_api.set_current_stream(cur_stream.handle)

    def __del__(self):
        cuda.destroy_stream(self.handle)

    def synchronize(self):
        cuda.stream_synchronize(self.handle)
