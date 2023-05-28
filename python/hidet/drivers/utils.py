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
import logging

import hidet.cuda

logger = logging.Logger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def lazy_initialize_cuda():
    # We intentionally query the cuda device information to put the properties of all devices to the lru_cache.
    #
    # Reasons:
    #   Hidet relies on the multiprocessing to parallelize the compilation. During the process, the forked process will
    #   query the properties of the device. If we do not cache the properties, the forked process will query the device
    #   via the cuda runtime API. However, the cuda runtime API does not work when the multiprocessing package is
    #   working in the fork mode. With the properties of all the GPUs cached, the forked process will not run any cuda
    #   runtime API and will not cause any problem.
    if getattr(lazy_initialize_cuda, '_initialized', False):
        return
    lazy_initialize_cuda._initialized = True  # pylint: disable=protected-access
    if hidet.cuda.available():
        for i in range(hidet.cuda.device_count()):
            hidet.cuda.properties(i)
            hidet.cuda.compute_capability(i)
