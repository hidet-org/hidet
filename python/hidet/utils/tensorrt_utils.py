from typing import List, Optional, Dict, Tuple
from collections import OrderedDict
from hashlib import sha256
import os
import json
import datetime
import tensorrt as trt
import hidet
from hidet.ffi import cuda
from hidet import Tensor
from hidet.utils import hidet_cache_dir
from hidet.testing import benchmark_func


class Profiler(trt.IProfiler):
    def __init__(self):
        super().__init__()
        self.layer2latency: Dict[str, float] = OrderedDict()

    def report_layer_time(self, layer_name, ms):
        self.layer2latency[layer_name] = ms

    def export_trace(self):
        from hidet.utils.profile_utils import TraceEvent

        events = []
        current_time = 0
        for layer, latency in self.layer2latency.items():
            events.append(TraceEvent(layer, 'op', 'B', current_time * 1000000, 0, 0, {'name': layer}))
            current_time += latency
            events.append(TraceEvent(layer, 'op', 'E', current_time * 1000000, 0, 0, {'name': layer}))
        return {'traceEvents': [event.export() for event in events], 'displayTimeUnit': 'ns'}


class Logger(trt.ILogger):
    def __init__(self, log_file: Optional[str] = None, print_out_level: str = 'INFO'):
        super().__init__()
        self.log_file = log_file
        self.print_out_level = print_out_level
        self.opened_file = None
        self.level_id = {'INTERNAL_ERROR': 0, 'ERROR': 1, 'WARNING': 2, 'INFO': 3, 'VERBOSE': 4}
        if self.log_file:
            self.opened_file = open(self.log_file, 'w')  # pylint: disable=consider-using-with

    def log(self, severity: trt.ILogger.Severity, msg: str):
        severity2name = {
            trt.ILogger.INTERNAL_ERROR: 'INTERNAL_ERROR',
            trt.ILogger.ERROR: 'ERROR',
            trt.ILogger.WARNING: 'WARNING',
            trt.ILogger.INFO: 'INFO',
            trt.ILogger.VERBOSE: 'VERBOSE',
        }
        severity_name = severity2name[severity]
        msg = '{} {} {}\n'.format(datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S'), severity_name, msg)
        self.opened_file.write(msg)
        # self.opened_file.flush()
        if self.level_id[self.print_out_level] >= self.level_id[severity_name] >= self.level_id['WARNING']:
            print(msg)
        if severity_name in ['INTERNAL_ERROR', 'ERROR']:
            raise RuntimeError('TensorRT ' + msg)

    def __del__(self):
        if self.opened_file:
            self.opened_file.close()
            self.opened_file = None


def milo_bytes(MiB):
    return MiB << 20


def create_engine_from_onnx(
    onnx_model_path: str,
    workspace_bytes: int = 512 << 20,
    input_shapes: Optional[Dict[str, List[int]]] = None,
    use_tf32: bool = False,
    use_fp16: bool = False,
) -> trt.ICudaEngine:
    cache_dir = hidet_cache_dir('trt_engine')
    os.makedirs(cache_dir, exist_ok=True)
    model_name = os.path.basename(onnx_model_path).split('.')[0]
    shape_hash = tuple((name, tuple(shape)) for name, shape in sorted(input_shapes.items(), key=lambda item: item[0]))
    shape_hash_suffix = sha256(str(shape_hash).encode()).hexdigest()[:6]
    engine_name = '{}{}{}_ws{}_{}.engine'.format(
        model_name,
        '_tf32' if use_tf32 else '',
        '_fp16' if use_fp16 else '',
        workspace_bytes // (1 << 20),
        shape_hash_suffix,
    )
    engine_path = os.path.join(cache_dir, engine_name)

    # logger = trt.Logger(min_severity=trt.Logger.ERROR)   # use WARNINGS when needed

    if os.path.exists(engine_path):
        # load the engine directly
        logger = Logger(engine_path + '.log', print_out_level='ERROR')
        runtime = trt.Runtime(logger)
        with open(engine_path, 'rb') as f:
            serialized_engine = f.read()
        engine = runtime.deserialize_cuda_engine(serialized_engine)
    else:
        build_logger = Logger(engine_path + '.build.log', print_out_level='ERROR')
        builder = trt.Builder(build_logger)
        # parse onnx model
        network: trt.INetworkDefinition = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        onnx_parser = trt.OnnxParser(network, build_logger)
        success = onnx_parser.parse_from_file(onnx_model_path)
        for idx in range(onnx_parser.num_errors):
            print(onnx_parser.get_error(idx))
        if not success:
            raise Exception('Failed parse onnx model in tensorrt onnx parser.')

        # set configs of the network builder
        config: trt.IBuilderConfig = builder.create_builder_config()
        # config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
        config.max_workspace_size = workspace_bytes
        # allow us to inspect the engine,
        # see https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#engine-inspector
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        # whether allow tf32/,
        # see https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#tf32-inference-c
        if use_tf32:
            config.set_flag(trt.BuilderFlag.TF32)
        else:
            config.clear_flag(trt.BuilderFlag.TF32)
        if use_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        else:
            config.clear_flag(trt.BuilderFlag.FP16)
        # force to use the precision in network definition,
        # see https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#layer-level-control
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)

        # optimization profiles required by dynamic inputs
        profile: trt.IOptimizationProfile = builder.create_optimization_profile()
        # assert len(inputs_shape) == network.num_inputs, 'Expect {} number of input shapes'.format(network.num_inputs)
        for i in range(network.num_inputs):
            tensor: trt.ITensor = network.get_input(i)
            if any(v == -1 for v in tensor.shape):
                if input_shapes is None or tensor.name not in input_shapes:
                    raise Exception(
                        "Found dynamic input: {}{}, ".format(tensor.name, list(tensor.shape))
                        + "please specify input_shapes as the target shape."
                    )
                opt_shape = input_shapes[tensor.name]
                profile.set_shape(tensor.name, min=opt_shape, opt=opt_shape, max=opt_shape)
        config.add_optimization_profile(profile)

        # build engine
        supported = builder.is_network_supported(network, config)
        if not supported:
            raise Exception('Network is not supported by TensorRT.')
        engine: trt.ICudaEngine = builder.build_engine(network, config)

        if engine is None:
            raise Exception('Can not build network with given config.')

        # save engine
        serialized_engine = builder.build_serialized_network(network, config)
        with open(engine_path, 'wb') as f:
            f.write(serialized_engine)
    return engine


dtype_map = {trt.DataType.INT32: 'int32', trt.DataType.FLOAT: 'float32', trt.DataType.HALF: 'float16'}


def _prepare_buffer(
    engine: trt.ICudaEngine, inputs: Dict[str, Tensor]
) -> Tuple[Dict[str, Tensor], Dict[str, Tensor], List[int]]:
    inputs = inputs.copy()
    outputs = {}
    buffers = []
    for i in range(engine.num_bindings):
        name = engine.get_binding_name(i)
        if engine.binding_is_input(i):
            dtype: trt.DataType = engine.get_binding_dtype(i)
            if name not in inputs:
                raise ValueError(
                    "TensorRT engine requires input '{}', but only received inputs: {}.".format(
                        name, list(inputs.keys())
                    )
                )
            if dtype != inputs[name].dtype.name:
                inputs[name] = hidet.graph.ops.cast(inputs[name], dtype_map[dtype])
            buffers.append(inputs[name].storage.addr)
        else:
            shape = engine.get_binding_shape(i)
            dtype: trt.DataType = engine.get_binding_dtype(i)
            output = hidet.empty(shape, dtype_map[dtype], device='cuda')
            outputs[name] = output
            buffers.append(output.storage.addr)
    return inputs, outputs, buffers


def engine_inference(engine: trt.ICudaEngine, inputs: Dict[str, Tensor]) -> Dict[str, Tensor]:
    # prepare inputs and outputs
    inputs, outputs, buffers = _prepare_buffer(engine, inputs)

    # inference
    context: trt.IExecutionContext = engine.create_execution_context()
    context.execute_async_v2(buffers, 0)
    cuda.device_synchronize()
    return outputs


def engine_benchmark(
    engine: trt.ICudaEngine, dummy_inputs: Dict[str, Tensor], warmup: int = 3, number: int = 5, repeat: int = 5
) -> List[float]:
    _, _, buffers = _prepare_buffer(engine, dummy_inputs)
    context: trt.IExecutionContext = engine.create_execution_context()
    return benchmark_func(lambda: context.execute_async_v2(buffers, 0), warmup, number, repeat, median=False)


def engine_inspect(engine: trt.ICudaEngine) -> Dict:
    inspector: trt.EngineInspector = engine.create_engine_inspector()
    layer_information = {}
    for i in range(engine.num_layers):
        layer_information['layer_{}'.format(i)] = json.loads(
            str(inspector.get_layer_information(i, trt.LayerInformationFormat.JSON))
        )
    # engine_information = json.loads(str(inspector.get_engine_information(trt.LayerInformationFormat.JSON)))
    return {
        'layers': layer_information,
        # 'engine': engine_information
    }


def engine_profiler(engine: trt.ICudaEngine, dummy_inputs: Dict[str, Tensor]) -> Dict:
    # prepare inputs and outputs
    _, _, buffers = _prepare_buffer(engine, dummy_inputs)
    context: trt.IExecutionContext = engine.create_execution_context()
    profiler = Profiler()
    context.profiler = profiler
    context.execute_v2(buffers)
    cuda.device_synchronize()
    return profiler.export_trace()
