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
from tabulate import tabulate
from hidet.cli.bench.model import BenchModel, all_registered_models, commonly_used_models
from .vision_model import VisionModel


resnet_models = {
    'resnet18': VisionModel('resnet18', 1, 3, 224, 224),
    'resnet34': VisionModel('resnet34', 1, 3, 224, 224),
    'resnet50': VisionModel('resnet50', 1, 3, 224, 224),
    'resnet101': VisionModel('resnet101', 1, 3, 224, 224),
    'resnet152': VisionModel('resnet152', 1, 3, 224, 224),
}


all_registered_models.extend(resnet_models.values())
commonly_used_models.append(resnet_models['resnet50'])


@click.command(name='resnet')
@click.option(
    '--models',
    type=str,
    default='resnet50',
    show_default=True,
    help='Comma seperated models to benchmark. Available models: {}'.format(', '.join(list(resnet_models.keys()))),
)
@click.option('-n', '--batch-size', default=1, show_default=True, help='Batch size')
@click.option('-c', '--channels', default=3, show_default=True, help='Input channels')
@click.option('-h', '--height', default=224, show_default=True, help='Input image height')
@click.option('-w', '--width', default=224, show_default=True, help='Input image width')
def bench_resnet(models: str, batch_size: int, channels: int, height: int, width: int):
    models = [model.strip() for model in models.split(',')]
    for model in models:
        if model not in resnet_models:
            raise ValueError('Unknown model: {}, candidates: {}'.format(model, list(resnet_models.keys())))

    bench_models = [VisionModel(model_name, batch_size, channels, height, width) for model_name in models]
    header = BenchModel.headers()
    result = [bench_model.benchmark() for bench_model in bench_models]

    click.echo(tabulate(result, headers=header, tablefmt='github', floatfmt='.3f', numalign='right', stralign='left'))
    click.echo('(PyTorch backend: allow_tf32={})'.format(BenchModel.allow_tf32))
