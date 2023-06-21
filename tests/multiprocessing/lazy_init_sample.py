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
import os


def main():
    import hidet

    ret = os.fork()
    if ret == 0:
        # child process
        p = 'child'
        a = hidet.randn([3, 4], device='cuda')
        print('I am the {} process, tensor: \n{}'.format(p, a))
    else:
        # parent process
        p = 'parent'
        a = hidet.randn([3, 4], device='cuda')
        _, child_ret = os.waitpid(ret, 0)
        print('I am the {} process, tensor: \n{}'.format(p, a))
        if child_ret != 0:
            raise RuntimeError('Child process exited with non-zero code: {}'.format(child_ret))


main()
