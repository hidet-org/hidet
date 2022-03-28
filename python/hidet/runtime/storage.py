from hidet.ffi.cuda_api import CudaAPI


class Storage:
    def __init__(self, addr):
        self.addr: int = addr

    def __del__(self):
        raise NotImplementedError()

    @staticmethod
    def new(device: str, num_bytes: int):
        if device == 'cuda':
            return CudaStorage(num_bytes)
        elif device == 'host' or device == 'cpu':
            return HostStorage(num_bytes)
        else:
            raise NotImplementedError()


class CudaStorage(Storage):
    def __init__(self, num_bytes: int):
        super().__init__(CudaAPI.malloc_async(num_bytes))
        self.num_bytes = num_bytes

    def __del__(self):
        CudaAPI.free_async(self.addr)


class HostStorage(Storage):
    def __init__(self, num_bytes: int):
        super().__init__(CudaAPI.malloc_host(num_bytes))
        self.num_bytes = num_bytes

    def __del__(self):
        CudaAPI.free_host(self.addr)
