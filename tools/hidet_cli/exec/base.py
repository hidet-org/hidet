from typing import List
import os
import click
import re
import hidet
from hidet.ffi.cuda_api import cuda
from hidet.testing import benchmark_func


class TensorArgType(click.ParamType):
    def __init__(self):
        super().__init__()
        self.p = re.compile(r'(\w+)\[([\d, ]*)]')

    name = 'tensor'

    def convert(self, value, param, ctx):
        if isinstance(value, str):
            match = self.p.match(value)
            if match is None:
                self.fail('invalid tensor format: {}'.format(value))
            match = self.p.match(value)
            dtype = match.group(1)
            shape_str = match.group(2)
            if dtype not in ['float32', 'float16', 'int64', 'int32', 'int8']:
                self.fail('invalid dtype: {}'.format(dtype))
            shape = [int(s) for s in shape_str.split(',')]
            if hidet.ir.ScalarType(dtype).is_float():
                return hidet.randn(shape, dtype=dtype)
            else:
                return hidet.zeros(shape, dtype=dtype)
        else:
            self.fail('invalid tensor format: {}'.format(value))


TensorArg = TensorArgType()


@click.command('exec')
@click.option('--opt', is_flag=True, default=False, help='Conduct graph-level optimizations (e.g., fusion and sub-graph rewrite).')
@click.option('--space', default='0', show_default=True, type=click.Choice(['0', '1', '2']), help='The search space of the tunable operator')
@click.option('--keep', is_flag=True, default=False, help='Keep the intermediate results of the graph optimizations.')
@click.option('--verbose', is_flag=True, default=False, help='Verbose output.')
@click.argument('model', type=click.Path(exists=True, dir_okay=False))
@click.argument('tensor', nargs=-1, type=TensorArg)
def exec_group(model: str, tensor: List[hidet.Tensor], opt: bool, space: str, keep: bool, verbose: bool):
    """
    Execute a model.

      MODEL: The model to execute, should be a path to a model file.
      Currently, only support onnx model (e.g., file like model.onnx).

      TENSOR: The arguments of the model. Can be a numpy tensor file .npy or a string represents the type and shape of a dummy input.

         The format of the string is like: "float32[1,3,224,224]" where the type is float32, and the shape is [1,3,224,224].
         Candidates of the type are: float32, float16, int64, int32, int16, int8, uint8.

    Examples

        \b
    hidet exec model.onnx float32[1,3,224,224] float32[1,1000]
    Run the model.onnx model with two inputs
    """
    module = hidet.graph.frontend.onnx.from_onnx(model)
    graph = module.flow_graph_for(tensor)

    hidet.space_level(int(space))

    model_name = os.path.basename(model).replace('.', '_')
    exec_name = model_name
    if opt:
        exec_name += '_opt'
    if space != 0:
        exec_name += '_s{}'.format(space)
    exec_dir = './outs/{}'.format(exec_name)

    if verbose and keep:
        click.echo('Output directory: {}'.format(exec_dir))

    if opt:
        click.echo('Optimizing the model...')
        with hidet.graph.PassContext() as ctx:
            if keep:
                ctx.save_graph_instrument(out_dir=os.path.join(exec_dir, 'graph-opt'))
            if verbose:
                ctx.profile_pass_instrument(print_stdout=True)
                ctx.set_verbose()
            graph = hidet.graph.optimize(graph)

    if keep:
        model_path = os.path.join(exec_dir, 'graph.hm')
        graph.save(model_path)
        click.echo('Graph saved to {}'.format(model_path))

    cuda_graph = graph.cuda_graph()
    cuda_graph.set_input_tensors(tensor)
    cuda_graph.run()
    cuda.device_synchronize()
    print('Latency: {:.3} ms'.format(benchmark_func(lambda: cuda_graph.run(), warmup=1, number=5, repeat=100, median=True)))
