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
from typing import Sequence
import numpy.testing
import torch
import torch.backends.cudnn
from torch import nn


class FunctionalModule(nn.Module):
    def __init__(self, op):
        super().__init__()
        self.op = op

    def forward(self, *args, **kwargs):
        return self.op(*args, **kwargs)


def check_module(model: torch.nn.Module, args: Sequence[torch.Tensor], atol=1e-4, rtol=1e-4, dynamic=False):
    model = model.cuda()
    model.eval()
    args = [x.cuda() if isinstance(x, torch.Tensor) else x for x in args]
    # we use a lambda to make sure the model is compiled by pytorch
    model_opt = torch.compile(lambda *args, **kwargs: model(*args, **kwargs), backend='hidet', dynamic=dynamic)

    torch.backends.cudnn.allow_tf32 = False  # disable tf32 for accuracy
    torch_outputs = model(*args)
    torch.backends.cudnn.allow_tf32 = True

    hidet_outputs = model_opt(*args)
    if isinstance(torch_outputs, torch.Tensor):
        torch_outputs = (torch_outputs,)
    if isinstance(hidet_outputs, torch.Tensor):
        hidet_outputs = (hidet_outputs,)

    if len(torch_outputs) != len(hidet_outputs):
        raise ValueError('torch_outputs and hidet_outputs have different length')

    for torch_output, hidet_output in zip(torch_outputs, hidet_outputs):
        torch_output = torch_output.detach().cpu().numpy()
        hidet_output = hidet_output.detach().cpu().numpy()
        numpy.testing.assert_allclose(torch_output, hidet_output, atol=atol, rtol=rtol)


# Class to initialise backend, run compilation
class Backend:
    def __init__(self, backend, dtype, search_space=2) -> None:
        assert backend in [
            'hidet',
            'max-autotune',
            'max-autotune-no-cudagraphs',
            'eager',
        ], 'backend is hidet or max-autotune or max-autotune-no-cudagraphs or eager supported only'
        self.backend = backend
        self.dtype = dtype
        self.search_space = search_space
        if self.backend == 'hidet':
            self.init_hidet()

    def init_hidet(self):
        import hidet
        import os

        use_fp16 = self.dtype == 'float16'
        hidet.torch.dynamo_config.search_space(self.search_space)
        hidet.torch.dynamo_config.use_fp16(use_fp16)
        hidet.torch.dynamo_config.use_fp16_reduction(use_fp16)
        hidet.torch.dynamo_config.use_attention(True)
        hidet.torch.dynamo_config.use_tensor_core(True)
        hidet.torch.dynamo_config.use_cuda_graph(True)
        hidet.option.search_space(self.search_space)

        # hidet.option.cache_dir(hidet.option.get_cache_dir() + '/regression')
        # hidet.option.parallel_tune(max_parallel_jobs=1)
        # hidet.option.debug_cache_tuning(True)
        # hidet.option.save_lower_ir(True)
        # hidet.option.debug_show_verbose_flow_graph(True)

        # Initialise compiler server
        if os.environ.get('CI_CS_HOSTNAME'):
            hidet.option.compile_server.addr(os.environ.get('CI_CS_HOSTNAME'))
            hidet.option.compile_server.port(int(os.environ.get('CI_CS_PORT')))
            hidet.option.compile_server.username(os.environ.get('CI_CS_USERNAME'))
            hidet.option.compile_server.password(os.environ.get('CI_CS_PASSWORD'))
            hidet.option.compile_server.repo(os.environ.get('REPO_NAME').strip(), os.environ.get('REPO_BRANCH').strip())
            hidet.option.compile_server.enable(flag=True)

    def compile(self, model):
        if self.backend == 'hidet':
            model = torch.compile(model, backend=self.backend)
        elif self.backend == 'eager':
            pass
        else:
            model = torch.compile(model, mode=self.backend)
        return model


# Make benchmarking of given torch model
def bench_torch_model(model, torch_inputs, bench_iters=100, warmup_iters=10):
    for _ in range(warmup_iters):
        out = model(*torch_inputs)  # pylint:disable=unused-variable
    torch.cuda.empty_cache()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(bench_iters):
        out = model(*torch_inputs)  # pylint:disable=unused-variable
    end.record()
    end.synchronize()
    torch.cuda.empty_cache()

    latency = start.elapsed_time(end) / bench_iters
    return latency
