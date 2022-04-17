import hidet
import os
import numpy as np
import onnx
import onnxruntime
from hidet.utils import Timer
from hidet.ffi import cuda_api


def run_onnx():
    model_path = os.path.expanduser('~/model_zoo/fused_unet_church.onnx')
    onnx_session = onnxruntime.InferenceSession(model_path)
    onnx_outputs = onnx_session.run(None, input_feed={
        'input.4': np.random.randn(1, 3, 256, 256).astype(np.float32),
        't.1': np.random.randn(1).astype(np.float32)
    })


def check():
    unet_onnx_path = os.path.expanduser('~/model_zoo/fused_unet_church.onnx')
    inputs = [np.random.randn(1, 3, 256, 256).astype(np.float32),
              np.random.randn(1).astype(np.float32)]

    # onnx
    onnx_session = onnxruntime.InferenceSession(unet_onnx_path)
    onnx_output, = onnx_session.run(None, input_feed={
        'input.4': inputs[0], 't.1': inputs[1]
    })

    # hidet
    hidet_model = hidet.tos.frontend.from_onnx(unet_onnx_path)
    hidet_inputs = [hidet.array(t).cuda() for t in inputs]
    hidet_output = hidet_model(*hidet_inputs)
    hidet_output = hidet_output.cpu().numpy()

    # check
    np.testing.assert_allclose(actual=hidet_output, desired=onnx_output, atol=1e-3, rtol=1e-3)


def main():
    unet_onnx_path = '~/model_zoo/fused_unet_church.onnx'
    model = hidet.tos.frontend.from_onnx(unet_onnx_path)
    inputs = [hidet.randn([1, 3, 256, 256], dtype='float32', device='cuda'),
              hidet.randn([1], dtype='float32', device='cuda')]
    outputs = model(*inputs)

    print(hidet.runtime.storage.cuda_pool)
    symbol_inputs = [hidet.symbol([1, 3, 256, 256], dtype='float32', device='cuda'),
                     hidet.symbol([1], dtype='float32', device='cuda')]
    symbol_outputs = model(*symbol_inputs)
    graph = hidet.trace_from(symbol_outputs, symbol_inputs)
    os.makedirs('./outs', exist_ok=True)
    with open('./outs/unet.json', 'w') as f:
        hidet.utils.netron.dump(graph, f)

    inputs = [hidet.empty_like(input) for input in symbol_inputs]
    for t in range(10):
        cuda_api.device_synchronization()
        with Timer('hidet traced {}'.format(t)):
            outputs = graph(*inputs)
            cuda_api.device_synchronization()
    print(hidet.runtime.storage.cuda_pool)

    with hidet.tos.transforms.PassContext(
        instruments=[
            hidet.tos.transforms.ProfileInstrument(print_stdout=True)
        ],
        verbose=False
    ):
        graph = hidet.tos.optimize(graph)
    os.makedirs('./outs', exist_ok=True)
    with open('./outs/unet_opt.json', 'w') as f:
        hidet.utils.netron.dump(graph, f)
    for t in range(10):
        cuda_api.device_synchronization()
        with Timer('hidet traced opt {}'.format(t)):
            outputs = graph(*inputs)
            cuda_api.device_synchronization()

    print(hidet.runtime.storage.cuda_pool)


if __name__ == '__main__':
    # check()
    # run_onnx()
    main()
