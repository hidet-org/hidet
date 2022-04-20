from typing import List, Optional, Tuple
import json
from tabulate import tabulate
import os
import time
import numpy as np
import argparse
import hidet as hi
import hidet
from hidet import Tensor
from hidet.utils import download, hidet_cache_dir, cuda, nvtx_annotate
from hidet.utils.git_utils import get_repo_sha, get_repo_commit_date
from hidet.ffi import cuda_api


def enviroment_info(args) -> str:
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


def onnx_model(name: str) -> str:
    if name == 'resnet50':
        model_path = download(
            url='https://media.githubusercontent.com/media/onnx/models/main/vision/classification/resnet/model/resnet50-v1-7.onnx',
            file_name='onnx/resnet50-v1-7.onnx',
            progress=True)
    elif name == 'bert-base-uncased':
        model_path = hi.utils.transformers_utils.export_transformer_model_as_onnx(
            model_name='bert-base-uncased',
            feature='default',
            output_dir=hidet_cache_dir(category='onnx')
        )
    else:
        raise NotImplementedError('Current do not support model {}.'.format(name))
    return model_path


def dummy_inputs(args) -> Tuple[List[str], List[Tensor]]:
    batch_size = args.bs
    if args.model == 'resnet50':
        names = ['data']
        tensors = [hi.randn([batch_size, 3, 224, 224], dtype='float32', device='cuda')]
        return names, tensors
    elif args.model == 'bert-base-uncased':
        vocab_size = 30522
        seq_length = args.bert_seq_length
        input_ids = hi.array(np.random.randint(0, vocab_size, [batch_size, seq_length], dtype=np.int64))
        attention_mask = hi.ones(shape=[batch_size, seq_length], dtype='int64')
        token_type_ids = hi.zeros(shape=[batch_size, seq_length], dtype='int64')
        names = ['input_ids', 'attention_mask', 'token_type_ids']
        tensors = [input_ids, attention_mask, token_type_ids]
        return names, tensors
    else:
        raise NotImplementedError('Model {}.'.format(args.model))


def benchmark_run(run_func, warmup, number, repeat) -> List[float]:
    results = []
    with nvtx_annotate('warmup'):
        for i in range(warmup):
            run_func()
            cuda_api.device_synchronization()
    for i in range(repeat):
        with nvtx_annotate(f'repeat {i}'):
            cuda_api.device_synchronization()
            start_time = time.time()
            for j in range(number):
                run_func()
            cuda_api.device_synchronization()
            end_time = time.time()
        results.append((end_time - start_time) * 1000 / number)
    return results


def bench_hidet(args, out_dir) -> List[float]:
    onnx_model_path: str = onnx_model(args.model)
    model = hidet.tos.frontend.onnx_utils.from_onnx(onnx_model_path)
    input_names, input_tensors = dummy_inputs(args)
    symbol_inputs = [hi.symbol_like(data) for data in input_tensors]
    outputs = model(*symbol_inputs)
    graph: hi.FlowGraph = hi.trace_from(outputs, inputs=symbol_inputs)
    with hidet.tos.PassContext(instruments=[
        hidet.tos.transforms.SaveGraphInstrument(out_dir=os.path.join(out_dir, 'ir'))  # dump ir
    ]):
        graph = hi.tos.transforms.optimize(graph)
    hidet.space_level(args.hidet_space)

    # dump trace
    hidet.utils.tracer.turn_on(True)
    graph(*input_tensors)
    hidet.utils.tracer.clear()
    graph(*input_tensors)
    trace = hidet.utils.tracer.export()
    with open(os.path.join(out_dir, 'trace.json'), 'w') as f:
        json.dump(trace, f)
    hidet.utils.tracer.turn_on(False)

    return benchmark_run(lambda: graph(*input_tensors), args.warmup, args.number, args.repeat)


def bench_trt(args, out_dir) -> List[float]:
    from hidet.utils.tensorrt_utils import create_engine_from_onnx, engine_benchmark, engine_inspect, engine_profiler
    onnx_model_path: str = onnx_model(args.model)
    input_names, input_tensors = dummy_inputs(args)
    engine = create_engine_from_onnx(onnx_model_path, inputs_shape={
        name: tensor.shape for name, tensor in zip(input_names, input_tensors)
    })
    dummy_inputs_dict = {name: tensor for name, tensor in zip(input_names, input_tensors)}
    results = engine_benchmark(
        engine=engine,
        dummy_inputs=dummy_inputs_dict,
        warmup=args.warmup, number=args.number, repeat=args.repeat
    )
    with open(os.path.join(out_dir, 'engine_inspect.json'), 'w') as f:
        json.dump(engine_inspect(engine), f, indent=2)
    with open(os.path.join(out_dir, 'engine_trace.json'), 'w') as f:
        json.dump(engine_profiler(engine, dummy_inputs_dict), f, indent=2)
    return results


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
    if args.exec == 'hidet':
        out_dir = os.path.join(out_dir, '{}_{}_space{}'.format(args.model, args.exec, args.hidet_space))
    else:
        out_dir = os.path.join(out_dir, '{}_{}'.format(args.model, args.exec))
    os.makedirs(out_dir, exist_ok=True)

    # bench
    bench_dict = {
        'hidet': bench_hidet,
        'trt': bench_trt
    }
    bench_func = bench_dict[args.exec]
    with nvtx_annotate(message=args.exec):
        results = bench_func(args, out_dir)

    # write results
    with open(os.path.join(out_dir, 'env.txt'), 'w') as f:
        f.write(enviroment_info(args))
    with open(os.path.join(out_dir, 'raw.json'), 'w') as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(out_dir, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    with open(os.path.join(out_dir, 'summary.txt'), 'w') as f:
        # model mode space median std
        head = '{:>10} {:>20} {:>12} {:>10} {:>10} {:>10}\n'.format(
            'BatchSize', 'Model', 'Executor', 'Space', 'Latency', 'Std'
        )
        summary = '{:>10} {:>20} {:>12} {:10} {:10.3f} {:10.3f}\n'.format(
            args.bs,
            args.model,
            args.exec,
            args.hidet_space,
            float(np.median(results)),
            float(np.std(results))
        )
        print(head + summary)
        f.write(head + summary)


parser = argparse.ArgumentParser(description='Hidet model benchmark script.')

# general parameters
parser.add_argument('--model', type=str, choices=['resnet50', 'bert-base-uncased'], required=True,
                    help='The model to benchmark.')
parser.add_argument('--exec', type=str, choices=['hidet', 'trt'], required=True,
                    help='Executor.')
parser.add_argument('--out_dir', type=str, default='./results/',
                    help='Output directory.')
parser.add_argument('--warmup', type=int, default=10, help='Number of warmups.')
parser.add_argument('--number', type=int, default=10, help='Number of runs per repeat.')
parser.add_argument('--repeat', type=int, default=10, help='Number of repeats.')

# executor parameters
# hidet executor parameters
parser.add_argument('--hidet_space', type=int, choices=[0, 1, 2], default=2,
                    help='The space level of each operator in the model. Large space level means longer compilation time and better performance.')

# model agnostic parameters
parser.add_argument('--bs', type=int, default=1, help='Batch size.')

# model specific parameters
# bert
parser.add_argument('--bert_seq_length', type=int, default=128, help='Sequence length of bert input.')
parser.add_argument('--bert_hidden_size', type=int, default=768, help='Hidden size of bert.')
parser.add_argument('--bert_vocab_size', type=int, default=30522, help='Vocabulary size of bert.')


if __name__ == '__main__':
    # main('--exec trt --model resnet50 --warmup 0 --number 1 --repeat 1')
    # main('--exec trt --model resnet50 --warmup 3 --number 5 --repeat 5')
    # main('--exec trt --model resnet50 --hidet_space 2 --warmup 3 --number 10 --repeat 10')
    # main('--exec hidet --model resnet50 --hidet_space 2 --warmup 3 --number 10 --repeat 10')
    main('--exec hidet --model bert-base-uncased --hidet_space 2 --warmup 3 --number 10 --repeat 10')
    # for model in ['resnet50', 'bert-base-uncased']:
    #     for exec in ['trt']:
    #         main(f'--exec {exec} --model {model} --number 10 --repeat 10')
    # for model in ['resnet50', 'bert-base-uncased']:
    #     for exec in ['trt', 'hidet']:
    #         main(f'--exec {exec} --model {model}')
