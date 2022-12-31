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
from hidet.cli.bench.model import commonly_used_models, all_registered_models
from hidet.utils import initialize
from .nlp_model import NLPModel, BenchModel


available_models = ['bert-base-uncased', 'bert-large-uncased', 'bert-base-cased', 'bert-large-cased', 'gpt2']

commonly_used_models.extend(
    [
        NLPModel('huggingface/pytorch-transformers', 'model', 'bert-base-uncased', 1, 128),
        NLPModel('huggingface/pytorch-transformers', 'model', 'gpt2', 1, 128),
    ]
)


@initialize()
def initialize_models():
    for model in available_models:
        for batch_size in [1, 8]:
            for seq_length in [128, 512]:
                all_registered_models.append(
                    NLPModel('huggingface/pytorch-transformers', 'model', model, batch_size, seq_length)
                )


@click.command(name='nlp')
@click.option(
    '--models',
    type=str,
    default='bert-base-uncased',
    show_default=True,
    help='Comma seperated models to benchmark. Available models: {}'.format(', '.join(available_models)),
)
@click.option('-n', '--batch-size', default=1, show_default=True, help='Batch size')
@click.option('-q', '--seq-length', default=128, show_default=True, help='Sequence length')
def bench_nlp(models: str, batch_size: int, seq_length: int):
    models = [model.strip() for model in models.split(',')]
    for model in models:
        if model not in available_models:
            raise ValueError('Unknown model: {}, candidates: {}'.format(model, list(available_models)))

    bench_models = [
        NLPModel('huggingface/pytorch-transformers', 'model', model, batch_size, seq_length) for model in models
    ]
    header = BenchModel.headers()
    result = [bench_model.benchmark() for bench_model in bench_models]

    click.echo(tabulate(result, headers=header, tablefmt='github', floatfmt='.3f', numalign='right', stralign='left'))
    click.echo('(PyTorch backend: allow_tf32={})'.format(BenchModel.allow_tf32))
