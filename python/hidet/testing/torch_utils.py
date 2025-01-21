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
import time
import numpy.testing
import torch
import torch.backends.cudnn
from torch import nn

import hidet
from hidet.utils import gc_disabled


class FunctionalModule(nn.Module):
    def __init__(self, op):
        super().__init__()
        self.op = op

    def forward(self, *args, **kwargs):
        return self.op(*args, **kwargs)


def check_module(
    model: torch.nn.Module, args: Sequence[torch.Tensor], device: str, atol=1e-4, rtol=1e-4, dynamic=False
):
    torch_device = device_to_torch(device)
    model = model.to(device_to_torch(device))
    model.eval()

    # convert args to the given device
    hidet_args = []
    torch_args = []
    for arg in args:
        if isinstance(arg, torch.Tensor):
            hidet_args.append(arg.clone().to(torch_device))
            torch_args.append(arg.clone().to(torch_device))
        else:
            assert not isinstance(arg, hidet.Tensor)
            hidet_args.append(arg)
            torch_args.append(arg)

    # we use a lambda to make sure the model is compiled by pytorch
    model_opt = torch.compile(
        lambda *args, **kwargs: model(*args, **kwargs), backend='hidet', mode=None, dynamic=dynamic
    )

    torch.backends.cudnn.allow_tf32 = False  # disable tf32 for accuracy
    torch_outputs = model(*torch_args)
    torch.backends.cudnn.allow_tf32 = True

    hidet_outputs = model_opt(*hidet_args)
    if isinstance(torch_outputs, torch.Tensor):
        torch_outputs = (torch_outputs,)
    if isinstance(hidet_outputs, torch.Tensor):
        hidet_outputs = (hidet_outputs,)

    if len(torch_outputs) != len(hidet_outputs):
        raise ValueError('torch_outputs and hidet_outputs have different length')

    for torch_output, hidet_output in zip(torch_outputs, hidet_outputs):
        # Turns out np.testing.assert_allclose sometimes can pass even if the shapes are different
        assert (
            torch_output.shape == hidet_output.shape
        ), f"Shape mismatch --- eager: {torch_output.shape} vs hidet: {hidet_output.shape}"
        assert (
            torch_output.dtype == hidet_output.dtype
        ), f"dtype mismatch --- eager: {torch_output.dtype} vs hidet: {hidet_output.dtype}"
        if torch_output.dtype == torch.bfloat16:
            torch_output = torch_output.to(torch.float32)
            hidet_output = hidet_output.to(torch.float32)
        torch_output = torch_output.detach().cpu().numpy()
        hidet_output = hidet_output.detach().cpu().numpy()
        numpy.testing.assert_allclose(torch_output, hidet_output, atol=atol, rtol=rtol)


# Class to initialise backend, run compilation
class Backend:
    def __init__(self, backend, mode, dtype, cache='') -> None:
        self.backend = backend
        self.mode = mode
        self.dtype = dtype
        self.cache = cache
        if self.backend == 'hidet':
            self.init_hidet()

    def init_hidet(self):
        import os

        hidet.torch.dynamo_config.use_tensor_core(True)
        hidet.option.cache_dir(hidet.option.get_cache_dir() + self.cache)
        hidet.torch.dynamo_config.steal_weights(True)
        # hidet.option.cache_dir(hidet.option.get_cache_dir() + '/regression')
        # hidet.option.num_local_workers(1)
        # hidet.option.debug_cache_tuning(True)
        # hidet.option.save_lower_ir(True)
        # hidet.option.debug_show_verbose_flow_graph(True)
        # hidet.torch.dynamo_config.dump_graph_ir("./graph_ir")

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
            model = torch.compile(model, backend='hidet', mode=self.mode)
        elif self.backend == 'eager':
            pass
        else:
            model = torch.compile(model, backend=self.backend, mode=self.mode)
        return model


# Make benchmarking of given torch model
def _bench_model(model, inputs, bench_iters, warmup_iters, true_outputs):
    for _ in range(warmup_iters):
        outs = model(*inputs)  # pylint:disable=unused-variable
    torch.cuda.empty_cache()

    if true_outputs is not None:
        torch.testing.assert_close(outs, true_outputs, rtol=0.2, atol=0.2)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()

    # import cProfile
    # profiler = cProfile.Profile()
    # profiler.enable()

    start = time.time_ns()
    for _ in range(bench_iters):
        _ = model(*inputs)  # pylint:disable=unused-variable
    torch.cuda.synchronize()
    end = time.time_ns()

    # profiler.disable()
    # profiler.dump_stats('10.prof')

    torch.cuda.empty_cache()

    latency = (end - start) / bench_iters / 10**6
    return latency


def bench_model(model, inputs, bench_iters=100, warmup_iters=10, true_outputs=None):
    with gc_disabled():
        return _bench_model(model, inputs, bench_iters, warmup_iters, true_outputs)


def _bench_gen_model(model, tokenizer, inputs, genlen, bench_iters, warmup_iters):
    assert genlen == 1  # This is really poor implementation. Will switch to vllm implementation
    END_OF_SENTENCE_ID = tokenizer.eos_token_id

    def one_iter(inputs):
        # text_output = ''
        for _ in range(genlen):
            outputs = model(inputs)
            logits = outputs.logits
            last_token_logits = logits[:, -1, :]
            probs = torch.softmax(last_token_logits, dim=-1)
            predicted_token_ids = torch.argmax(probs, dim=-1)
            if any(id == END_OF_SENTENCE_ID for id in predicted_token_ids):
                break
            # predicted_text = tokenizer.decode(predicted_token_ids[0])
            # text_output += predicted_text + ' '

            predicted_token_ids = torch.reshape(predicted_token_ids, (1, 1))
            inputs = torch.cat([inputs, predicted_token_ids], dim=1)

        # print(text_output)
        return inputs

    # torch._dynamo.mark_dynamic(inputs, 0)  # pylint: disable=protected-access
    for _ in range(warmup_iters):
        output_text = one_iter(inputs)
    torch.cuda.empty_cache()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(bench_iters):
        output_text = one_iter(inputs)
    end.record()
    end.synchronize()
    torch.cuda.empty_cache()

    latency = start.elapsed_time(end) / bench_iters
    return latency, output_text


def bench_gen_model(model, tokenizer, inputs, bs=1, genlen=1, bench_iters=3, warmup_iters=1):
    with gc_disabled():
        return _bench_gen_model(model, tokenizer, inputs, genlen, bench_iters, warmup_iters)


def device_to_torch(hidet_device: str) -> str:
    if hidet_device in ['cuda', 'hip']:
        return 'cuda'
    elif hidet_device == 'cpu':
        return 'cpu'
    else:
        raise NotImplementedError(hidet_device)
