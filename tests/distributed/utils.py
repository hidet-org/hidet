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
import pytest
import multiprocessing
from multiprocessing import Process
import os
import pytest

import hidet
import hidet.distributed

TMP_PATH = './tmp'


def distributed_test(world_size):
    def decorator(func):
        def _f():
            if os.path.exists(TMP_PATH):
                os.remove(TMP_PATH)

            def proc(i):
                hidet.cuda.set_device(i)
                hidet.distributed.init_process_group(init_method=f'file://{TMP_PATH}', world_size=world_size, rank=i)
                hidet.distributed.set_nccl_comms()
                func(i)

            processes = [Process(target=proc, args=(i,)) for i in range(world_size)]
            for p in processes:
                p.start()
            for p in processes:
                p.join()

        return pytest.mark.skip(reason='This test requires multiple GPUs')(_f)

    return decorator
