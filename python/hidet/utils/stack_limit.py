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
"""
Allow more stack space and recursion depth for python.
"""
from typing import Tuple
import warnings
import sys
import resource

# allow up to 512 MiB stack space
expected_stack_size = 2**29  # 512 MiB
stack_limit: Tuple[int, int] = resource.getrlimit(resource.RLIMIT_STACK)
if stack_limit[1] != resource.RLIM_INFINITY and stack_limit[1] < expected_stack_size:
    warnings.warn(
        f'The hard limit for stack size is too small ({stack_limit[1] / 2**20:.1f} MiB), '
        f'we recommend to increase it to {expected_stack_size / 2**20:.1f} MiB. '
        'If you are the root user on Linux OS, you could refer to `man limits.conf` to increase this limit.'
    )
    resource.setrlimit(resource.RLIMIT_STACK, (stack_limit[1], stack_limit[1]))
else:
    resource.setrlimit(resource.RLIMIT_STACK, (expected_stack_size, stack_limit[1]))

# allow up to 10^5 recursive python calls, increase this when needed
sys.setrecursionlimit(100000)
