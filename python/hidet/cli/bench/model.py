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
# pylint: disable=ungrouped-imports, no-name-in-module
from typing import List, Any, Optional
import click
from hidet.testing import benchmark_func
import hidet


class BenchModel:
    search_space = 0
    dtype = None  # torch.float32
    tensor_core: bool = False
    disable_torch_cudnn_tf32 = False
    enable_torch_cublas_tf32 = False
    report_path: Optional[str] = None

    def __str__(self):
        raise NotImplementedError()

    def model(self):
        """
        Returns a pytorch model to benchmark.

        Returns
        -------
        ret: torch.nn.Module
            The model to benchmark.
        """
        raise NotImplementedError()

    def example_inputs(self):
        """
        Returns a tuple of (args, kwargs) as the inputs to pass to the model.

        Returns
        -------
        ret: Tuple[Sequence[torch.Tensor], Dict[str, torch.Tensor]]
            The inputs to pass to the model.
        """
        raise NotImplementedError()

    def converted_model(self):
        import torch

        if BenchModel.dtype is None:
            BenchModel.dtype = torch.float32
        model = self.model().eval()
        model = model.to(dtype=BenchModel.dtype)
        model = model.cuda()
        return model

    def converted_inputs(self):
        import torch

        args, kwargs = self.example_inputs()

        def convert_f32(arg):
            if arg.dtype == torch.float32:
                return arg.to(dtype=BenchModel.dtype)
            return arg

        args = [convert_f32(arg.cuda()) for arg in args]
        kwargs = {k: convert_f32(v.cuda()) for k, v in kwargs.items()}
        return args, kwargs

    @staticmethod
    def tensor_str(tensor) -> str:
        """
        Returns a string representation of a tensor.

        Parameters
        ----------
        tensor: torch.Tensor
            The tensor to represent.

        Returns
        -------
        ret: str
            The string representation of the tensor.
        """
        dtype = hidet.torch.utils.dtype_from_torch(tensor.dtype)
        return '{}[{}]'.format(dtype.short_name, ','.join(map(str, tensor.shape)))

    def inputs_str(self) -> str:
        """
        Returns a string representation of the inputs to the model.

        Returns
        -------
        ret: str
            The string representation of the inputs to the model.
        """
        args, kwargs = self.converted_inputs()
        items = []
        for arg in args:
            items.append(self.tensor_str(arg))
        for k, v in kwargs.items():
            items.append('{}={}'.format(k, self.tensor_str(v)))
        return ', '.join(items)

    @classmethod
    def report_table(cls, table_str):
        if cls.report_path is not None:
            with open(cls.report_path, 'w') as f:
                click.echo(table_str, file=f)
        click.echo(table_str)

    def bench_with_backend(self, backend: str, mode=None, warmup=3, number=10, repeat=10):
        try:
            import torch.backends.cudnn  # pylint: disable=redefined-outer-name
            import torch.backends.cuda

            if not hidet.torch.dynamo_available():
                raise RuntimeError('Torch Dynamo is not available, please install pytorch 2.0 or higher.')
            import torch._dynamo as dynamo

            torch.backends.cudnn.allow_tf32 = not self.disable_torch_cudnn_tf32
            torch.backends.cuda.matmul.allow_tf32 = self.enable_torch_cublas_tf32

            model, (args, kwargs) = self.converted_model(), self.converted_inputs()
            dynamo.reset()
            with torch.no_grad():
                model_opt = torch.compile(model, backend=backend, mode=mode)
                latency = benchmark_func(
                    run_func=lambda: model_opt(*args, **kwargs), warmup=warmup, number=number, repeat=repeat
                )
            return latency
        except Exception as e:  # pylint: disable=broad-except
            from traceback import format_exc

            print('Failed to benchmark {} with {}: {}\nTraceback:\n{}'.format(self, backend, e, format_exc()))
            return float('NaN')

    def bench_eager(self) -> float:
        print('Benchmarking {} with backend {}...'.format(self, 'eager'))
        return self.bench_with_backend('eager')

    def bench_inductor(self, mode: str) -> float:
        print('Benchmarking {} with backend {}...'.format(self, 'inductor(mode={})'.format(mode)))
        return self.bench_with_backend('inductor', mode=mode)

    def bench_hidet(self, use_cuda_graph=True, use_fp16=False, use_fp16_reduction=False) -> float:
        print('Benchmarking {} with backend {}...'.format(self, 'hidet(space={})'.format(self.search_space)))
        config = hidet.torch.dynamo_config
        config.search_space(self.search_space)
        config.use_cuda_graph(use_cuda_graph)
        config.use_tensor_core(self.tensor_core)
        config.use_fp16(use_fp16)
        config.use_fp16_reduction(use_fp16_reduction)
        return self.bench_with_backend('hidet')

    @staticmethod
    def headers() -> List[str]:
        return [
            'model',
            'inputs',
            'eager',
            'reduce-overhead',
            'max-autotune',
            'hidet({})'.format(BenchModel.search_space),
        ]

    def benchmark(self) -> List[Any]:
        return [
            str(self),
            self.inputs_str(),
            self.bench_eager(),
            self.bench_inductor('reduce-overhead'),
            self.bench_inductor('max-autotune'),
            self.bench_hidet(),
        ]


all_registered_models: List[BenchModel] = []
commonly_used_models: List[BenchModel] = []
