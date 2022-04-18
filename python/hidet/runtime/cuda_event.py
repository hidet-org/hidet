from typing import List
from hidet.ffi import cuda_api


class CudaEvent:
    def __init__(self, handle, pool):
        self.handle: int = handle
        self.pool: CudaEventPool = pool

    def __del__(self):
        self.pool.event_handles.append(self.handle)
        self.handle = 0

    def elapsed_time_since(self, start_event) -> float:
        return cuda_api.event_elapsed_time(start_event.handle, self.handle)

    def record_on(self, stream_handle: int = 0):
        cuda_api.event_record(self.handle, stream_handle)


class CudaEventPool:
    def __init__(self):
        self.event_handles: List[int] = []

    def new_event(self) -> CudaEvent:
        if len(self.event_handles) > 0:
            return CudaEvent(self.event_handles.pop(), self)
        else:
            return CudaEvent(cuda_api.create_event(), self)

    def __del__(self):
        while len(self.event_handles) > 0:
            handle = self.event_handles.pop()
            cuda_api.destroy_event(handle)


cuda_event_pool = CudaEventPool()
