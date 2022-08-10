from bench.common import BenchResult, get_onnx_model


def bench_tvm(args, out_dir) -> BenchResult:
    from hidet.utils.tvm_utils import tvm_graph_module_from_onnx, tvm_benchmark, tvm_inference
    result = BenchResult()

    # configs
    result.configs = 'trial_{}'.format(args.tvm_trial)

    # latencies
    onnx_path, input_names, input_tensors = get_onnx_model(name=args.model, batch_size=args.bs, precision=args.precision)
    gmod = tvm_graph_module_from_onnx(
        onnx_model_path=onnx_path,
        input_shapes={
            name: tensor.shape for name, tensor in zip(input_names, input_tensors)
        },
        tune_autotvm=(args.exec == 'autotvm'),
        tune_ansor=(args.exec == 'ansor'),
        tune_trial_per_task=args.tvm_trial
    )
    inputs = {name: tensor for name, tensor in zip(input_names, input_tensors)}
    result.latencies = tvm_benchmark(
        gmod,
        dummy_inputs=inputs,
        warmup=args.warmup, number=args.number, repeat=args.repeat
    )

    # outputs
    result.outputs = tvm_inference(gmod, inputs)

    return result


