from bench.common import BenchResult, get_onnx_model


def bench_ort(args, out_dir) -> BenchResult:
    from hidet.utils.ort_utils import create_ort_session, ort_benchmark, ort_inference
    result = BenchResult()

    # configs
    result.configs = 'provider_{}'.format(args.ort_provider)
    provider = {
        'cuda': 'CUDAExecutionProvider',
        'trt': 'TensorrtExecutionProvider'
    }[args.ort_provider]

    # latencies
    onnx_path, input_names, input_tensors = get_onnx_model(name=args.model, batch_size=args.bs, precision=args.precision)
    session = create_ort_session(onnx_path, provider=provider)
    inputs = {name: tensor for name, tensor in zip(input_names, input_tensors)}
    result.latencies = ort_benchmark(
        session,
        dummy_inputs=inputs,
        warmup=args.warmup, number=args.number, repeat=args.repeat
    )

    # outputs
    result.outputs = list(ort_inference(session, inputs=inputs).values())

    return result


