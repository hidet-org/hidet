from typing import List
import json
from tabulate import tabulate
import os
import time
import numpy as np
import argparse
import hidet as hi
from hidet import Tensor
from hidet.utils import download, hidet_cache_dir, cuda
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


def dummy_inputs(args) -> List[Tensor]:
    batch_size = args.bs
    if args.model == 'resnet50':
        return [hi.randn([batch_size, 3, 224, 224], dtype='float32', device='cuda')]
    elif args.model == 'bert-base-uncased':
        vocab_size = 30522
        seq_length = args.bert_seq_length
        input_ids = hi.array(np.random.randint(0, vocab_size, [batch_size, seq_length], dtype=np.int64))
        attention_mask = hi.ones(shape=[batch_size, seq_length], dtype='int64')
        token_type_ids = hi.zeros(shape=[batch_size, seq_length], dtype='int64')
        return [input_ids, attention_mask, token_type_ids]
    else:
        raise NotImplementedError('Model {}.'.format(args.model))


def main(args):
    # load model
    model_path = onnx_model(args.model)
    model = hi.tos.frontend.onnx_utils.from_onnx(model_path)

    # prepare inputs
    inputs = dummy_inputs(args)

    # prepare run function
    results = []
    if args.mode == 'imperative':
        run = lambda: model(*inputs)
    elif args.mode.startswith('lazy'):
        symbol_inputs = [hi.symbol_like(data) for data in inputs]
        outputs = model(*symbol_inputs)
        graph: hi.FlowGraph = hi.trace_from(outputs)
        if args.mode == 'lazy_opt':
            graph = hi.tos.transforms.optimize(graph)
        run = lambda: graph(*inputs)
    else:
        raise NotImplementedError()

    # measure latency
    warmup, number, repeat = args.warmup, args.number, args.repeat
    for i in range(warmup):
        run()
    for i in range(repeat):
        cuda_api.device_synchronization()
        start_time = time.time()
        for j in range(number):
            run()
        end_time = time.time()
        cuda_api.device_synchronization()
        results.append((end_time - start_time) * 1000 / number)

    # write results
    out_dir = os.path.join(args.out_dir, '{}_space{}_{}'.format(args.model, args.space, args.mode))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'env.txt'), 'w') as f:
        f.write(enviroment_info(args))
    with open(os.path.join(out_dir, 'raw.json'), 'w') as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(out_dir, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    with open(os.path.join(out_dir, 'summary.txt'), 'w') as f:
        # model mode space median std
        summary = '{:>30} {:>12} {} {:.3f} {:.3f}\n'.format(
            args.model,
            args.mode,
            args.space,
            float(np.median(results)),
            float(np.std(results))
        )
        f.write(summary)


parser = argparse.ArgumentParser(description='Hidet model benchmark script.')

# general parameters
parser.add_argument('--model', type=str, choices=['resnet50', 'bert-base-uncased'], required=True,
                    help='The model to benchmark.')
parser.add_argument('--space', type=int, choices=[0, 1, 2], default=0,
                    help='The space level of each operator in the model. '
                         'Large space level means longer compilation time and better performance.')
parser.add_argument('--mode', type=str, choices=['imperative', 'lazy', 'lazy_opt'], default='lazy',
                    help='The execution mode, can be imperative mode and lazy mode. '
                         'Lazy mode will perform graph-level optimizations.')
parser.add_argument('--out_dir', type=str, default='./results/', help='Output directory.')
parser.add_argument('--warmup', type=int, default=3, help='Number of warmups.')
parser.add_argument('--number', type=int, default=5, help='Number of runs per repeat.')
parser.add_argument('--repeat', type=int, default=5, help='Number of repeats.')

# model agnostic parameters
parser.add_argument('--bs', type=int, default=1, help='Batch size.')

# model specific parameters
# bert
parser.add_argument('--bert_seq_length', type=int, default=128, help='Sequence length of bert input.')
parser.add_argument('--bert_hidden_size', type=int, default=768, help='Hidden size of bert.')
parser.add_argument('--bert_vocab_size', type=int, default=30522, help='Vocabulary size of bert.')

if __name__ == '__main__':
    args = parser.parse_args()
    args.out_dir = os.path.join(args.out_dir,
                                '{}_{}'.format(get_repo_commit_date(), get_repo_sha(short=True)),
                                cuda.query_device_name(short=True),
                                'models')
    main(args)
