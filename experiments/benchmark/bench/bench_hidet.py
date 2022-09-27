import os
import os
import time

import hidet
from bench.common import benchmark_run, BenchResult, get_onnx_model
from hidet.utils import hidet_cache_file


def bench_hidet(args, out_dir) -> BenchResult:
    result = BenchResult()
    # print('args', args, 'time stamp', time.time())

    # configs
    result.configs = 'sp{}_{}_{}_{}_pk_{}'.format(args.hidet_space, args.mma, args.precision, args.reduce_precision, args.parallel_k)

    # latencies and outputs
    graph_path = hidet_cache_file(
        'hidet_graph',
        args.model,
        'bs_{}_{}'.format(args.bs, result.configs),
        'graph.pickle'
    )
    onnx_path, input_names, input_tensors = get_onnx_model(name=args.model, batch_size=args.bs, precision=args.precision)

    hidet.space_level(args.hidet_space)

    if os.path.exists(graph_path) and not args.disable_graph_cache:
        graph = hidet.load_graph(graph_path)
    else:
        if args.disable_graph_cache:
            print('disabled graph cache, rebuilding...')
        t1 = time.time()
        model = hidet.graph.frontend.onnx.from_onnx(onnx_path)
        symbol_inputs = [hidet.symbol_like(data) for data in input_tensors]
        outputs = model(*symbol_inputs)
        graph: hidet.FlowGraph = hidet.trace_from(outputs, inputs=symbol_inputs)
        with hidet.graph.PassContext() as ctx:
            short2long = {
                'f16': 'float16',
                'f32': 'float32',
                'bf16': 'bfloat16'
            }
            ctx.save_graph_instrument(out_dir=os.path.join(out_dir, 'ir'))
            ctx.set_precision(short2long[args.precision])
            ctx.set_reduce_precision(short2long[args.reduce_precision])
            ctx.set_mma(args.mma)
            if args.parallel_k == 'disabled':
                ctx.set_parallel_k(disabled=True)
            elif args.parallel_k == 'default':
                ctx.set_parallel_k(default=True)
            elif args.parallel_k == 'search':
                ctx.set_parallel_k(search=True)
            else:
                ctx.set_parallel_k(nparts=int(args.parallel_k))

            graph = hidet.graph.transforms.optimize(graph)

        hidet.save_graph(graph, graph_path + '.tmp')
        os.rename(graph_path + '.tmp', graph_path)

        graph(*input_tensors)
        t2 = time.time()
        with open(os.path.join(os.path.dirname(graph_path), 'tuning_time.txt'), 'w') as f:
            f.write(str((t2 - t1) / 60.0) + ' minutes')

    cuda_graph = graph.cuda_graph()
    result.outputs = cuda_graph.run_with_inputs(input_tensors)
    result.latencies = benchmark_run(lambda: cuda_graph.run(), args.warmup, args.number, args.repeat)

    return result
