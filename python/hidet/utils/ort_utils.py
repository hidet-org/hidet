from time import time
from typing import Dict, List

import onnxruntime as ort

import hidet
from hidet import Tensor
from hidet.ffi import cuda_api


def create_ort_session(onnx_model_path) -> ort.InferenceSession:
    session = ort.InferenceSession(onnx_model_path, providers=['CUDAExecutionProvider'])
    session.disable_fallback()
    return session


def _prepare_io_binding(session: ort.InferenceSession, inputs: Dict[str, Tensor]) -> ort.IOBinding:
    input_values: Dict[str, ort.OrtValue] = {
        name: ort.OrtValue.ortvalue_from_numpy(tensor.numpy(), device_type='cuda') for name, tensor in inputs.items()
    }
    output_names = [output.name for output in session.get_outputs()]
    io_binding = session.io_binding()
    for name, value in input_values.items():
        io_binding.bind_ortvalue_input(name, value)
    for name in output_names:
        io_binding.bind_output(name, device_type='cuda')
    return io_binding


def ort_inference(session: ort.InferenceSession, inputs: Dict[str, Tensor]) -> Dict[str, Tensor]:
    io_binding = _prepare_io_binding(session, inputs)
    session.run_with_iobinding(iobinding=io_binding)
    outputs = {output_node.name: hidet.array(value.numpy()).cuda()
               for output_node, value in zip(session.get_outputs(), io_binding.get_outputs())}
    return outputs


def ort_benchmark(session: ort.InferenceSession, dummy_inputs: Dict[str, Tensor], warmup=10, number=10, repeat=10) -> List[float]:
    io_binding = _prepare_io_binding(session, dummy_inputs)
    for i in range(warmup):
        session.run_with_iobinding(iobinding=io_binding)
    results = []
    for i in range(repeat):
        cuda_api.device_synchronization()
        start_time = time()
        for j in range(number):
            session.run_with_iobinding(iobinding=io_binding)
        cuda_api.device_synchronization()
        end_time = time()
        results.append((end_time - start_time) * 1000 / number)
    return results


if __name__ == '__main__':
    model_path = hidet.utils.hidet_cache_file('onnx', 'resnet50-v1-7.onnx')
    session = create_ort_session(model_path)
    inputs = {
        'data': hidet.randn([1, 3, 224, 224])
    }
    outputs = ort_inference(session, inputs)
    print(outputs)
    print(ort_benchmark(session, inputs))
