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
from typing import Optional, Callable, Dict, List, Union
import os
import time
import inspect
import numpy as np
import hidet
from hidet.graph import Tensor
from hidet.graph.ir.flow_graph import FlowGraph
from hidet.graph.tensor import symbol_like


def get_type_repr(value):
    if isinstance(value, (str, int, float)):
        return str(type(value).__name__)
    elif isinstance(value, list):
        items = [get_type_repr(v) for v in value]
        return '[{}]'.format(', '.join(items))
    elif isinstance(value, tuple):
        items = [get_type_repr(v) for v in value]
        return '({})'.format(', '.join(items))
    elif isinstance(value, dict):
        for v in value.keys():
            if not isinstance(v, str):
                raise TypeError('Only support str as dict key, got {}'.format(type(v)))
        keys = list(v for v in value.keys())
        items = [get_type_repr(v) for v in value.values()]
        return '{{{}}}'.format(', '.join('{}: {}'.format(k, v) for k, v in zip(keys, items)))
    elif isinstance(value, Tensor):
        shape_repr = ', '.join(str(v) for v in value.shape)
        return '{}:{}[{}]'.format(value.device, value.dtype.name, shape_repr)
    elif isinstance(value, np.ndarray):
        shape_repr = ', '.join(str(v) for v in value.shape)
        return 'np.{}[{}]'.format(value.dtype, shape_repr)
    else:
        raise TypeError('Does not support type {} for jit.'.format(type(value)))


def get_bind_repr(bind: inspect.BoundArguments) -> str:
    items = []
    for name, value in bind.arguments:
        items += '{}: {}'.format(name, get_type_repr(value))
    return 'BindRepr({})'.format(', '.join(items))


class JitGraph:
    # todo: use inspect package to support more wide range input and outputs
    def __init__(
        self,
        func: Callable,
        opt: bool = False,
        parallel_k: str = 'default',
        save_ir_dir: Optional[str] = './outs',
        mma: str = 'wmma_tf32_f32',
    ):
        self.func: Callable = func
        self.cached_graph: Dict[str, FlowGraph] = {}

        self.parallel_k = parallel_k
        self.opt = opt
        self.save_ir_dir = os.path.join(save_ir_dir, func.__name__)
        self.mma = mma

    def __str__(self):
        items = []
        for args_repr, graph in self.cached_graph.items():
            items.extend([args_repr, ' => ', str(graph), '\n'])
        return ''.join(items)

    @staticmethod
    def args_representation(*args):
        args_repr = get_type_repr(args)
        return args_repr

    def flow_graph_for(self, *args) -> FlowGraph:
        args_repr = self.args_representation(*args)

        if args_repr not in self.cached_graph:
            symbol_inputs = [symbol_like(arg) if isinstance(arg, Tensor) else arg for arg in args]
            symbol_outputs = self.func(*symbol_inputs)
            graph = hidet.trace_from(symbol_outputs, inputs=[v for v in symbol_inputs if isinstance(v, Tensor)])
            if self.opt:
                with hidet.graph.PassContext() as ctx:
                    ctx.save_graph_instrument(self.save_ir_dir)
                    ctx.set_mma(self.mma)
                    if self.parallel_k == 'default':
                        ctx.set_parallel_k(default=True)
                    elif self.parallel_k == 'disabled':
                        ctx.set_parallel_k(disabled=True)
                    else:
                        ctx.set_parallel_k(nparts=int(self.parallel_k))
                    graph = hidet.graph.optimize(graph)
            self.cached_graph[args_repr] = graph
        graph: FlowGraph = self.cached_graph[args_repr]
        return graph

    def __call__(self, *args):
        graph = self.flow_graph_for(*args)
        return graph(*args)

    def benchmark(self, *args, warmup=10, number=10, repeat=10, median=True) -> Union[float, List[float]]:
        graph = self.flow_graph_for(*args)
        cuda_graph = graph.cuda_graph()
        cuda_graph.set_input_tensors(args)

        results = []
        for _ in range(warmup):
            cuda_graph.run()
            hidet.cuda.synchronize()
        for _ in range(repeat):
            hidet.cuda.synchronize()
            start_time = time.time()
            for _ in range(number):
                cuda_graph.run()
            hidet.cuda.synchronize()
            end_time = time.time()
            results.append((end_time - start_time) * 1000 / number)

        if median:
            return float(np.median(results))
        else:
            return results


def jit(opt=False, save_ir_dir='./outs', parallel_k='default', mma='simt') -> Callable[[Callable], JitGraph]:
    def decorator(func) -> JitGraph:
        jit_graph = JitGraph(func=func, opt=opt, parallel_k=parallel_k, save_ir_dir=save_ir_dir, mma=mma)
        return jit_graph

    return decorator
