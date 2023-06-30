from dataclasses import dataclass


@dataclass
class CudaComputeCapability:
    maxThreadsPerBlock: int
    sharedMemPerBlock: int


_compute_capability = {
    'sm_70': CudaComputeCapability(1024, 96 * 1024),
    'sm_72': CudaComputeCapability(1024, 96 * 1024),
    'sm_75': CudaComputeCapability(1024, 64 * 1024),
    'sm_80': CudaComputeCapability(1024, 163 * 1024),
    'sm_86': CudaComputeCapability(1024, 99 * 1024),
    'sm_87': CudaComputeCapability(1024, 163 * 1024),
    'sm_89': CudaComputeCapability(1024, 99 * 1024),
    'sm_90': CudaComputeCapability(1024, 227 * 1024),
}


def capability(arch: str) -> CudaComputeCapability:
    if arch not in _compute_capability:
        raise ValueError(f'Unsupported CUDA architecture {arch}')
    return _compute_capability[arch]
