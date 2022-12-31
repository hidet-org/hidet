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
import click
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
    '--torch-tf32',
    default=False,
    show_default=True,
    type=bool,
    help='Enable torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32.',
)
def bench_group(space: str, torch_tf32: bool):
    BenchModel.search_space = int(space)
    BenchModel.allow_tf32 = torch_tf32


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
        bench_group.add_command(command)
