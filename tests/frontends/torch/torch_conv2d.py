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
import torch
from hidet.testing.torch_utils import check_module
import torch.backends.cudnn as cudnn


# @pytest.mark.parametrize(
#     'in_shape,w_shape,stride,padding',
#     [
#         # [[1, 3, 224, 224], [42, 3, 7, 7], 2, 3],
#         [[1, 3, 224, 224], [42, 3, 7, 7], 3, 4]
#     ],
# )
# @pytest.mark.parametrize('groups', [
#     1,
#     3
# ])
# @pytest.mark.parametrize('dtype', [torch.float32])
def demo_conv2d(in_shape, w_shape, stride, padding, groups, dtype):
    cudnn.allow_tf32 = False
    check_module(
        model=torch.nn.Conv2d(
            in_channels=in_shape[1],
            out_channels=w_shape[0],
            kernel_size=w_shape[2:],
            stride=stride,
            padding=padding,
            groups=groups,
        ),
        args=[torch.randn(in_shape, dtype=dtype)],
    )
    cudnn.allow_tf32 = True


if __name__ == '__main__':
    # pytest.main([__file__])
    import hidet
    import os

    hidet.option.cache_dir(os.path.join(hidet.option.get_cache_dir(), 'test_cache'))
    print('Cache directory: {}'.format(hidet.option.get_cache_dir()))

    hidet.option.compile_server.addr('23.21.86.254')
    hidet.option.compile_server.port(3281)
    hidet.option.compile_server.username('admin')
    hidet.option.compile_server.password('admin_password')
    # hidet.option.compile_server.repo('hidet-org/hidet', 'main')
    hidet.option.compile_server.repo('yaoyaoding/hidet', 'fix-cserver')
    hidet.option.compile_server.enable()
    demo_conv2d(in_shape=[1, 3, 224, 224], w_shape=[42, 3, 7, 7], stride=2, padding=3, groups=1, dtype=torch.float32)
    demo_conv2d(in_shape=[1, 3, 224, 224], w_shape=[42, 3, 7, 7], stride=2, padding=3, groups=3, dtype=torch.float32)

