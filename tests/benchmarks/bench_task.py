import argparse
import hidet

from hidet.runtime.compiled_task import CompiledTask
from hidet.drivers import build_task
from hidet.testing.torch_utils import bench_model
from hidet.testing.utils import init_hidet


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Benchmark task')
    parser.add_argument('--task', type=str, default='./task.pickle', help='Path to dump of task')
    parser.add_argument('--cache', type=str, default='', help='')
    args = parser.parse_args()
    task_path, cache = args.task, args.cache

    init_hidet(cache=cache)

    task: hidet.Task = hidet.load_task(task_path)
    inputs = task.dummy_arguments('cuda')
    compiled_task: CompiledTask = build_task(task, target='cuda')

    # For dynamic shapes should set their value
    for tensor in task.params:
        for dim in tensor.shape:
            if isinstance(dim, hidet.ir.expr.SymbolVar):
                hidet.ffi.runtime_api.set_symbol_value(dim.name, 2)

    out = compiled_task(*inputs)

    lat = bench_model(compiled_task, inputs)

    print(lat)
