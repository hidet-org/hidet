from typing import List, Optional, Dict
import os
import tensorrt as trt
import hidet
from hidet import Tensor, randn, empty
from hidet.utils import hidet_cache_dir


def milo_bytes(MiB):
    return MiB << 20


def create_engine_from_onnx(onnx_model_path: str, workspace_bytes: int = 512 << 20, inputs_shape: Optional[Dict[str, List[int]]] = None) -> trt.ICudaEngine:
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)

    cache_dir = hidet_cache_dir('trt_engine')
    os.makedirs(cache_dir, exist_ok=True)
    model_name = os.path.basename(onnx_model_path).split('.')[0]
    engine_name = '{}_ws{}.engine'.format(model_name, workspace_bytes // (1 << 20))
    engine_path = os.path.join(cache_dir, engine_name)

    if os.path.exists(engine_path):
        # load the engine directly
        runtime = trt.Runtime(logger)
        with open(engine_path, 'rb') as f:
            serialized_engine = f.read()
        engine = runtime.deserialize_cuda_engine(serialized_engine)
    else:
        # parse onnx model
        network: trt.INetworkDefinition = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        onnx_parser = trt.OnnxParser(network, logger)
        success = onnx_parser.parse_from_file(onnx_model_path)
        for idx in range(onnx_parser.num_errors):
            print(onnx_parser.get_error(idx))
        if not success:
            raise Exception('Failed parse onnx model in tensorrt onnx parser.')

        # set configs of the network builder
        config: trt.IBuilderConfig = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)

        # optimization profiles required by dynamic inputs
        profile: trt.IOptimizationProfile = builder.create_optimization_profile()
        # assert len(inputs_shape) == network.num_inputs, 'Expect {} number of input shapes'.format(network.num_inputs)
        for i in range(network.num_inputs):
            tensor: trt.ITensor = network.get_input(i)
            if any(v == -1 for v in tensor.shape):
                if inputs_shape is None or tensor.name not in inputs_shape:
                    raise Exception("Found dynamic input: {}{}, "
                                    "please specify input_shapes as the target shape.".format(tensor.name, list(tensor.shape)))
                opt_shape = inputs_shape[tensor.name]
                profile.set_shape(tensor.name, min=opt_shape, opt=opt_shape, max=opt_shape)
        config.add_optimization_profile(profile)

        # build engine
        supported = builder.is_network_supported(network, config)
        if not supported:
            raise Exception('Network is not supported by TensorRT.')
        engine: trt.ICudaEngine = builder.build_engine(network, config)

        # save engine
        serialized_engine = builder.build_serialized_network(network, config)
        with open(engine_path, 'wb') as f:
            f.write(serialized_engine)
    return engine


def engine_inference(engine: trt.ICudaEngine, inputs: List[Tensor]) -> List[Tensor]:
    dtype_map = {
        trt.DataType.INT32: 'int32',
        trt.DataType.FLOAT: 'float32',
    }
    print('num bindings: ', engine.num_bindings)
    for i in range(engine.num_bindings):
        print('Binding {}: {} {}'.format(i, engine.get_binding_dtype(i).name, engine.get_binding_name(i)))
    outputs = []
    for i in range(engine.num_bindings):
        if engine.binding_is_input(i):
            dtype: trt.DataType = engine.get_binding_dtype(i)
            if dtype != inputs[i].dtype:
                inputs[i] = hidet.tos.operators.cast(inputs[i], dtype)
        else:
            shape = engine.get_binding_shape(i)
            dtype: trt.DataType = engine.get_binding_dtype(i)
            outputs.append(hidet.empty(shape, dtype_map[dtype], device='cuda'))
    buffers = [tensor.storage.addr for tensor in inputs + outputs]
    return outputs


def engine_benchmark(engine: trt.ICudaEngine, warmup: int = 3, number: int = 5, repeat: int = 5) -> List[float]:
    pass


if __name__ == '__main__':
    # onnx_model_path = os.path.join(hidet_cache_dir('onnx'), 'resnet50-v1-7.onnx')
    onnx_model_path = os.path.join(hidet_cache_dir('onnx'), 'bert-base-uncased.onnx')
    engine = create_engine_from_onnx(onnx_model_path, inputs_shape={
        'input_ids': [1, 512],
        'attention_mask': [1, 512],
        'token_type_ids': [1, 512]
    })
    x = randn([1, 3, 224, 224], dtype='float32', device='cuda')
    outputs = engine_inference(engine, [x])

