# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Optional
from dataclasses import dataclass


@dataclass
class CudaComputeCapability:
    maxThreadsPerBlock: int
    sharedMemPerBlock: int
    regsPerBlock: int
    regsPerMultiprocessor: int


# the numbers are from CUDA Programming Guide
# see: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications
_compute_capability = {
    'sm_70': CudaComputeCapability(
        maxThreadsPerBlock=1024, sharedMemPerBlock=96 * 1024, regsPerBlock=64 * 1024, regsPerMultiprocessor=64 * 1024
    ),
    'sm_72': CudaComputeCapability(
        maxThreadsPerBlock=1024, sharedMemPerBlock=96 * 1024, regsPerBlock=64 * 1024, regsPerMultiprocessor=64 * 1024
    ),
    'sm_75': CudaComputeCapability(
        maxThreadsPerBlock=1024, sharedMemPerBlock=64 * 1024, regsPerBlock=64 * 1024, regsPerMultiprocessor=64 * 1024
    ),
    'sm_80': CudaComputeCapability(
        maxThreadsPerBlock=1024, sharedMemPerBlock=163 * 1024, regsPerBlock=64 * 1024, regsPerMultiprocessor=64 * 1024
    ),
    'sm_86': CudaComputeCapability(
        maxThreadsPerBlock=1024, sharedMemPerBlock=99 * 1024, regsPerBlock=64 * 1024, regsPerMultiprocessor=64 * 1024
    ),
    'sm_87': CudaComputeCapability(
        maxThreadsPerBlock=1024, sharedMemPerBlock=163 * 1024, regsPerBlock=64 * 1024, regsPerMultiprocessor=64 * 1024
    ),
    'sm_89': CudaComputeCapability(
        maxThreadsPerBlock=1024, sharedMemPerBlock=99 * 1024, regsPerBlock=64 * 1024, regsPerMultiprocessor=64 * 1024
    ),
    'sm_90': CudaComputeCapability(
        maxThreadsPerBlock=1024, sharedMemPerBlock=227 * 1024, regsPerBlock=64 * 1024, regsPerMultiprocessor=64 * 1024
    ),
    'sm_90a': CudaComputeCapability(
        maxThreadsPerBlock=1024, sharedMemPerBlock=227 * 1024, regsPerBlock=64 * 1024, regsPerMultiprocessor=64 * 1024
    ),
    'sm_100': CudaComputeCapability(
        maxThreadsPerBlock=1024, sharedMemPerBlock=227 * 1024, regsPerBlock=64 * 1024, regsPerMultiprocessor=64 * 1024
    ),
    'sm_100a': CudaComputeCapability(
        maxThreadsPerBlock=1024, sharedMemPerBlock=227 * 1024, regsPerBlock=64 * 1024, regsPerMultiprocessor=64 * 1024
    ),
}


def capability(arch: Optional[str] = None) -> CudaComputeCapability:
    """
    Get the information of the compute capability of a CUDA architecture.

    Parameters
    ----------
    arch: str, optional
        The CUDA architecture. If not specified, the current CUDA architecture (`hidet.option.cuda.get_arch()`) will be
        used.

    Returns
    -------
    capability: CudaComputeCapability
        The compute capability of the specified CUDA architecture.
    """
    if arch is None:
        import hidet.option

        arch = hidet.option.cuda.get_arch()
    if arch not in _compute_capability:
        raise ValueError(f'Unsupported CUDA architecture {arch}')
    return _compute_capability[arch]
