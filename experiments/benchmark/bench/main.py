from typing import List, Optional, Tuple, Union
import json
from tabulate import tabulate
import os
import numpy as np
import argparse
import hidet
from hidet.utils import cuda, nvtx_annotate, hidet_cache_file, error_tolerance
from hidet.utils.git_utils import get_repo_sha, get_repo_commit_date
from bench.common import BenchResult, get_onnx_model, run_with_onnx
from .bench_ort import bench_ort
from .bench_hidet import bench_hidet
from .bench_tvm import bench_tvm
from .bench_trt import bench_trt
from .bench_torch import bench_torch
from .bench_tf import bench_tf
from .bench_manual import bench_manual


parser = argparse.ArgumentParser(description='Hidet model benchmark script.')

# ======

# general parameters
parser.add_argument('--model', type=str,
                    # choices=['resnet50', 'inception_v3', 'mobilenet_v2', 'bert', 'bart'],
                    required=True,
                    help='The model to benchmark.')
parser.add_argument('--exec', type=str, choices=['hidet', 'trt', 'ort', 'tvm', 'autotvm', 'ansor', 'tf', 'tf_xla', 'torch', 'manual'], required=True,
                    help='Executor.')
parser.add_argument('--out_dir', type=str, default='./results/',
                    help='Output directory.')
parser.add_argument('--nocheck', action='store_true', help='Do not check the output. Used when profiling each single kernel.')
parser.add_argument('--warmup', type=int, default=10, help='Number of warmups.')
parser.add_argument('--number', type=int, default=10, help='Number of runs per repeat.')
parser.add_argument('--repeat', type=int, default=10, help='Number of repeats.')

# ======

# executor parameters
# hidet executor parameters
parser.add_argument('--precision', choices=['f16', 'bf16', 'f32'], default='f32')
parser.add_argument('--reduce_precision', choices=['f16', 'f32'], default='f32')
parser.add_argument('--mma', choices=['simt', 'wmma', 'mma', 'mma_custom'], default='simt')
parser.add_argument('--hidet_space', type=int, choices=[0, 1, 2], default=2,
                    help='The space level of each operator in the model. Large space level means longer compilation time and better performance.')
parser.add_argument('--parallel_k', choices=['disabled', 'default', 'search', '2', '4', '6', '8'], default='default')
parser.add_argument('--disable-graph-cache', action='store_true')

# tvm number of trial per task
parser.add_argument('--tvm_trial', type=int, default=800, help='Number of trial per task in autotvm and ansor, default 800.')

# tensorrt configs
parser.add_argument('--trt_tf32', action='store_true')
parser.add_argument('--trt_fp16', action='store_true')

# onnx runtime configs
parser.add_argument('--ort_provider', choices=['cuda', 'trt'], default='cuda')

# hidet manual config
parser.add_argument('--manual_config', type=str, default='default', help='Custom config for hidet manual executor')

# ======

# model agnostic parameters
parser.add_argument('--bs', type=int, default=1, help='Batch size.')

# model specific parameters
# bert
parser.add_argument('--bert_seq_length', type=int, default=128, help='Sequence length of bert input.')
parser.add_argument('--bert_hidden_size', type=int, default=768, help='Hidden size of bert.')
parser.add_argument('--bert_vocab_size', type=int, default=30522, help='Vocabulary size of bert.')


def environment_info(args) -> str:
    return str(tabulate(
        headers=[
            'Name', 'Value'
        ],
        tabular_data=[
            ['Commit', get_repo_sha()],
            ['GPU', cuda.query_device_name()],
            ['Arch', cuda.query_arch()],
            ['Compute Capacity', cuda.query_compute_capability()],
            ['Current SM Clock (MHz)', cuda.query_gpu_current_clock()],
            ['Current Memory Clock (MHz)', cuda.query_memory_current_clock()],
            ['Warmup/Number/Repeat', '{} / {} / {}'.format(args.warmup, args.number, args.repeat)]
        ]
    ))


def main(command_line_args: Optional[str] = None):
    if command_line_args:
        args = parser.parse_args(command_line_args.strip().split())
    else:
        args = parser.parse_args()
    # output dir
    out_dir = os.path.join(args.out_dir,
                           '{}_{}'.format(get_repo_commit_date(), get_repo_sha(short=True)),
                           cuda.query_device_name(short=True),
                           'models')
    exec_name = 'bs{}_{}_{}_{}_{}'.format(args.bs, args.model, args.exec, args.precision, args.reduce_precision)
    if args.exec == 'hidet':
        exec_name += '_space{}_pk_{}'.format(args.hidet_space, args.parallel_k)
    elif args.exec in ['autotvm', 'ansor']:
        exec_name += '_trial{}'.format(args.tvm_trial)
    out_dir = os.path.join(out_dir, exec_name)
    os.makedirs(out_dir, exist_ok=True)

    # bench
    bench_dict = {
        'hidet': bench_hidet,
        'trt': bench_trt,
        'ort': bench_ort,
        'autotvm': bench_tvm,
        'ansor': bench_tvm,
        'tvm': bench_tvm,
        'tf': bench_tf,
        'tf_xla': bench_tf,
        'torch': bench_torch,
        'manual': bench_manual
    }
    bench_func = bench_dict[args.exec]
    with nvtx_annotate(message=args.exec):
        with hidet.utils.py.Timer() as timer:
            result: BenchResult = bench_func(args, out_dir)

    # error tolerance
    et = -1.0
    if not args.nocheck:
        onnx_path, input_names, input_tensors = get_onnx_model(name=args.model, batch_size=args.bs, precision=args.precision)
        onnx_outputs = run_with_onnx(model_path=onnx_path, input_names=input_names, input_tensors=input_tensors)
        if result.outputs is not None:
            for baseline_output, onnx_output in zip(result.outputs, onnx_outputs):
                et = max(et, error_tolerance(baseline_output.numpy(), onnx_output))


    # write results
    with open(os.path.join(out_dir, 'env.txt'), 'w') as f:
        f.write(environment_info(args))
    with open(os.path.join(out_dir, 'raw.json'), 'w') as f:
        raw = {
            'latency': result.latencies,
            'bench_time': timer.elapsed_seconds()
        }
        json.dump(raw, f, indent=2)
    with open(os.path.join(out_dir, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    with open(os.path.join(out_dir, 'summary.txt'), 'w') as f:
        # model mode space median std
        head = '{:>10} {:>20} {:>12} {:>40} {:>10} {:>10} {:>10} {:>10}\n'.format(
            'BatchSize', 'Model', 'Executor', 'Config', 'Space', 'Latency', 'Std', 'Error'
        )
        summary = '{:>10} {:>20} {:>12} {:>40} {:10} {:10.3f} {:10.3f} {}\n'.format(
            args.bs,
            args.model,
            args.exec,
            result.configs,
            args.hidet_space,
            float(np.median(result.latencies)),
            float(np.std(result.latencies)),
            f'{et:10.3f}' if et != -1.0 else '{:>10}'.format('N/A')
        )
        print(head + summary)
        f.write(head + summary)
