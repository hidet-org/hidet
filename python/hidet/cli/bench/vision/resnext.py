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


resnext_models = {
    'resnext50_32x4d': VisionModel('resnext50_32x4d', 1, 3, 224, 224),
    'resnext101_32x8d': VisionModel('resnetxt101_32x8d', 1, 3, 224, 224),
}


all_registered_models.extend(resnext_models.values())
commonly_used_models.append(resnext_models['resnext50_32x4d'])


@click.command(name='resnext')
@click.option(
    '--models',
    type=str,
    default='resnext50_32x4d',
    show_default=True,
    help='Comma seperated models to benchmark. Available models: {}'.format(', '.join(list(resnext_models.keys()))),
)
@click.option('-n', '--batch-size', default=1, show_default=True, help='Batch size')
@click.option('-c', '--channels', default=3, show_default=True, help='Input channels')
@click.option('-h', '--height', default=224, show_default=True, help='Input image height')
@click.option('-w', '--width', default=224, show_default=True, help='Input image width')
def bench_resnext(models: str, batch_size: int, channels: int, height: int, width: int):
    models = [model.strip() for model in models.split(',')]
    for model in models:
        if model not in resnext_models:
            raise ValueError('Unknown model: {}, candidates: {}'.format(model, list(resnext_models.keys())))

    bench_models = [VisionModel(model_name, batch_size, channels, height, width) for model_name in models]
    header = BenchModel.headers()
    result = [bench_model.benchmark() for bench_model in bench_models]

    click.echo(tabulate(result, headers=header, tablefmt='github', floatfmt='.3f', numalign='right', stralign='left'))
    click.echo('(PyTorch backend: allow_tf32={})'.format(BenchModel.allow_tf32))
