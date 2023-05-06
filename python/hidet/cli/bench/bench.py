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
import click
import hidet
from hidet.utils import initialize
from . import vision
from . import nlp
from .model import BenchModel
from .bench_common import bench_common
from .bench_all import bench_all


@click.group(name='bench', help='Benchmark models.')
@click.option(
    '--space',
    default='0',
    show_default=True,
    type=click.Choice(['0', '1', '2']),
    help='Schedule space. 0: default schedule. 1: small schedule space. 2: large schedule space.',
)
@click.option(
    '--dtype', default='float32', show_default=True, type=click.Choice(['float32', 'float16']), help='Data type to use.'
)
@click.option(
    '--tensor-core',
    default=False,
    show_default=True,
    is_flag=True,
    type=bool,
    help='Whether to use tensor core in hidet.',
)
@click.option('--report', type=click.Path(exists=False, dir_okay=False, writable=True), help='Report file path.')
@click.option(
    '--disable-torch-cudnn-tf32',
    default=False,
    is_flag=True,
    type=bool,
    help='Set torch.backends.cudnn.allow_tf32=False.',
)
@click.option(
    '--enable-torch-cublas-tf32',
    default=False,
    is_flag=True,
    type=bool,
    help='Set torch.backends.cuda.matmul.allow_tf32=True.',
)
@click.option(
    '--cache-dir',
    default=None,
    type=click.Path(dir_okay=True, file_okay=False, writable=True),
    help='The cache directory to store the generated kernels.',
)
def hidet_bench_group(
    space: str,
    dtype: str,
    tensor_core: bool,
    report: Optional[click.Path],
    disable_torch_cudnn_tf32: bool,
    enable_torch_cublas_tf32: bool,
    cache_dir: Optional[click.Path],
):
    import torch

    BenchModel.search_space = int(space)
    BenchModel.dtype = getattr(torch, dtype)
    BenchModel.tensor_core = tensor_core
    BenchModel.disable_torch_cudnn_tf32 = disable_torch_cudnn_tf32
    BenchModel.enable_torch_cublas_tf32 = enable_torch_cublas_tf32
    BenchModel.report_path = report
    if cache_dir:
        hidet.option.cache_dir(str(cache_dir))


@initialize()
def register_commands():
    for command in [
        bench_common,
        bench_all,
        vision.bench_resnet,
        vision.bench_resnext,
        vision.bench_inception_v3,
        vision.bench_mobilenet_v2,
        nlp.bench_nlp,
    ]:
        assert isinstance(command, click.Command)
        hidet_bench_group.add_command(command)
