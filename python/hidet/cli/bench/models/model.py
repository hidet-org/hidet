# pylint: disable=ungrouped-imports, no-name-in-module
from typing import Tuple, Sequence, Dict, List, Any
import torch
import torch.backends.cudnn
from torch import nn
from hidet.testing import benchmark_func
import hidet

if hidet.torch.dynamo_available():
    import torch._dynamo as dynamo
else:
    dynamo = None

torch.backends.cudnn.allow_tf32 = False  # for fair comparison


class BenchModel:
    def __str__(self):
        raise NotImplementedError()

    def model(self) -> nn.Module:
        raise NotImplementedError()

    def example_inputs(self) -> Tuple[Sequence[torch.Tensor], Dict[str, torch.Tensor]]:
        raise NotImplementedError()

    @staticmethod
    def tensor_str(tensor: torch.Tensor) -> str:
        dtype = hidet.torch.utils.dtype_from_torch(tensor.dtype)
        return '{}[{}]'.format(dtype.short_name, ','.join(map(str, tensor.shape)))

    def inputs_str(self) -> str:
        args, kwargs = self.example_inputs()
        items = []
        for arg in args:
            items.append(self.tensor_str(arg))
        for k, v in kwargs.items():
            items.append('{}={}'.format(k, self.tensor_str(v)))
        return ', '.join(items)

    def bench_with_backend(self, backend: str, warmup=3, number=10, repeat=10):
        model, (args, kwargs) = self.model(), self.example_inputs()
        model = model.cuda().eval()
        args = [arg.cuda() for arg in args]
        kwargs = {k: v.cuda() for k, v in kwargs.items()}
        dynamo.reset()
        with torch.no_grad():
            model_opt = torch.compile(model, backend=backend)
            latency = benchmark_func(
                run_func=lambda: model_opt(*args, **kwargs), warmup=warmup, number=number, repeat=repeat
            )
        return latency

    def bench_eager(self) -> float:
        return self.bench_with_backend('eager')

    def bench_inductor(self) -> float:
        return self.bench_with_backend('inductor')

    def bench_hidet(self, use_cuda_graph=True, use_fp16=False, use_fp16_reduction=False, space=2) -> float:
        config = hidet.torch.dynamo_config
        config.search_space(space)
        config.use_cuda_graph(use_cuda_graph)
        config.use_fp16(use_fp16)
        config.use_fp16_reduction(use_fp16_reduction)
        return self.bench_with_backend('hidet')

    @staticmethod
    def headers() -> List[str]:
        return ['model', 'inputs', 'eager', 'inductor', 'hidet', 'hidet_f16']

    def benchmark(self) -> List[Any]:
        return [
            str(self),
            self.inputs_str(),
            self.bench_eager(),
            self.bench_inductor(),
            self.bench_hidet(),
            self.bench_hidet(use_fp16=True),
        ]


all_registered_models: List[BenchModel] = []
commonly_used_models: List[BenchModel] = []
