import os
import json
from bench.common import BenchResult, get_onnx_model


def bench_trt(args, out_dir) -> BenchResult:
    from hidet.utils.tensorrt_utils import create_engine_from_onnx, engine_benchmark, engine_inspect, engine_profiler, engine_inference
    result = BenchResult()

    # configs
    configs = []
    if args.trt_fp16:
        configs.append('fp16')
    if args.trt_tf32:
        configs.append('tf32')
    if len(configs) == 0:
        configs.append('fp32')
    result.configs = '_'.join(configs)

    # latencies
    onnx_path, input_names, input_tensors = get_onnx_model(name=args.model, batch_size=args.bs, precision=args.precision)
    engine = create_engine_from_onnx(
        onnx_model_path=onnx_path,
        input_shapes={name: tensor.shape for name, tensor in zip(input_names, input_tensors)},
        workspace_bytes=512 << 20,  # 512 MiB
        use_tf32=args.trt_tf32,
        use_fp16=args.trt_fp16
    )
    dummy_inputs_dict = {name: tensor for name, tensor in zip(input_names, input_tensors)}
    result.latencies = engine_benchmark(
        engine=engine,
        dummy_inputs=dummy_inputs_dict,
        warmup=args.warmup, number=args.number, repeat=args.repeat
    )

    # outputs
    result.outputs = list(engine_inference(engine, inputs=dummy_inputs_dict).values())

    # extra information
    with open(os.path.join(out_dir, 'engine_inspect.json'), 'w') as f:
        json.dump(engine_inspect(engine), f, indent=2)
    with open(os.path.join(out_dir, 'engine_trace.json'), 'w') as f:
        json.dump(engine_profiler(engine, dummy_inputs_dict), f, indent=2)

    return result
