from bench.common import BenchResult, benchmark_run


def bench_torch(args, out_dir) -> BenchResult:
    from hidet.testing.torch_models.all import get_torch_model
    result = BenchResult()
    model, input_dict = get_torch_model(args.model, batch_size=args.bs)

    def run_func():
        model(**input_dict)

    result.latencies = benchmark_run(run_func, warmup=args.warmup, number=args.number, repeat=args.repeat)
    result.configs = 'fp32'
    result.outputs = None
    return result

