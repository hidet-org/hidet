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
class HIPCComputeCapability:
    maxThreadsPerBlock: int
    warpSize: int
    regsPerBlock: int
    sharedMemPerBlock: int
    major: int
    minor: int
    gcnArchName: str


# TODO: add more architectures
#     I haven't found a resource online that documents this information, these numbers were gotten from C++ code
#     eg. hipDeviceProp_t (perhaps this could be automated in the future)
_compute_capability = {
    'gfx1100': HIPCComputeCapability(
        maxThreadsPerBlock=1024,
        warpSize=32,
        regsPerBlock=65536,
        sharedMemPerBlock=65536,
        major=11,
        minor=0,
        gcnArchName='gfx1100',
    ),
    'gfx90a': HIPCComputeCapability(
        maxThreadsPerBlock=1024,
        warpSize=64,
        regsPerBlock=65536,
        sharedMemPerBlock=65536,
        major=9,
        minor=0,
        gcnArchName='gfx90a',
    ),
}


def capability(arch: Optional[str] = None) -> HIPCComputeCapability:
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

        # TODO: support HIP
        arch = hidet.option.hip.get_arch()
    if arch not in _compute_capability:
        raise ValueError(f'Unsupported CUDA architecture {arch}')
    return _compute_capability[arch]
