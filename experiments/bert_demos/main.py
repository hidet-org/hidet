import os
import numpy as np
import onnx
from hidet.ffi import cuda_api
import onnxruntime
import hidet
import hidet as hi
from hidet import tos
from hidet.utils import hidet_cache_dir, Timer


def demo_old_resnet50():
    import onnx
    import onnxruntime
    model_path = '/home/yaoyao/model_zoo/resnet50_v1.onnx'
    # model_path = '/home/yaoyao/model_zoo/resnet50-v2-7.onnx'
    model = onnx.load_model(model_path)
    onnx.checker.check_model(model)
    # run use hidet
    x_hidet = hi.randn([1, 3, 224, 224])
    module = hidet.tos.frontend.onnx_utils.OnnxModule(model)
    y_hidet = module(x_hidet)
    # run use onnx runtime
    onnx_infer = onnxruntime.InferenceSession(model_path)
    y_onnx = onnx_infer.run(None, {'input_tensor:0': x_hidet.cpu().numpy()})
    print(y_hidet[1])
    print(y_onnx[1])
    np.testing.assert_allclose(actual=y_hidet[1].cpu().numpy(), desired=y_onnx[1], rtol=1e-5, atol=1e-5)


def demo_resnet50():
    import numpy as np
    from hidet import randn
    from hidet.tos.frontend.onnx_utils import OnnxModule
    from hidet.utils import download

    import onnx
    import onnxruntime
    model_path = download(
        url='https://media.githubusercontent.com/media/onnx/models/main/vision/classification/resnet/model/resnet50-v1-7.onnx',
        file_name='onnx/resnet50-v1-7.onnx',
        progress=True)
    model = onnx.load_model(model_path)
    onnx.checker.check_model(model)
    # run use hidet
    # x_hidet = randn([1, 3, 224, 224], device='cuda')
    x_hidet = hi.zeros([1, 3, 224, 224], device='cuda')
    module = OnnxModule(model)
    y_hidet = module(x_hidet)
    y_hidet = y_hidet.cpu().numpy()

    x_symbol = hi.symbol([1, 3, 224, 224], dtype='float32', device='cuda')
    y_symbol = module(x_symbol)
    graph = hi.trace_from(y_symbol)
    with open('./outs/graph.json', 'w') as f:
        hi.utils.netron.dump(graph, f)

    # run use onnx runtime
    onnx_infer = onnxruntime.InferenceSession(model_path)
    y_onnx = onnx_infer.run(None, {'data': x_hidet.cpu().numpy()})
    y_onnx = y_onnx[0]

    # compare
    np.testing.assert_allclose(actual=y_hidet, desired=y_onnx, rtol=1e-5, atol=1e-5)


def demo_bert():
    model_path = hidet.utils.transformers_utils.export_transformer_model_as_onnx(
        'bert-base-uncased', feature='default', output_dir=hidet.utils.hidet_cache_dir('onnx')
    )

    batch_size = 1
    seq_length = 512
    vocab_size = 30522
    input_ids = np.random.randint(0, vocab_size, [batch_size, seq_length], dtype=np.int64)
    attention_mask = np.ones(shape=[batch_size, seq_length], dtype=np.int64)
    token_type_ids = np.zeros(shape=[batch_size, seq_length], dtype=np.int64)

    # onnx
    onnx_session = onnxruntime.InferenceSession(model_path)
    onnx_outputs = onnx_session.run(None, input_feed={
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'token_type_ids': token_type_ids
    })
    onnx_last_hidden_state, onnx_pooler_output = onnx_outputs
    print(onnx_last_hidden_state.shape)

    # hidet
    hidet_model = tos.frontend.from_onnx(model_path)
    print(hidet_model.input_names)
    print(hidet_model.output_names)
    input_ids = hidet.array(input_ids).cuda()
    attention_mask = hidet.array(attention_mask).cuda()
    token_type_ids = hidet.array(token_type_ids).cuda()
    hidet_outputs = hidet_model(input_ids, attention_mask, token_type_ids)
    hidet_outputs = [out.cpu().numpy() for out in hidet_outputs]
    last_hidden_state, pooler_output = hidet_outputs
    np.testing.assert_allclose(actual=hidet_outputs[0], desired=onnx_outputs[0], rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(actual=hidet_outputs[1], desired=onnx_outputs[1], rtol=1e-5, atol=1e-5)

    # hidet lazy
    inputs = [input_ids, attention_mask, token_type_ids]
    symbol_inputs = [hi.symbol_like(t) for t in inputs]
    symbol_outputs = hidet_model(*symbol_inputs)
    graph = hi.trace_from(symbol_outputs, inputs=symbol_inputs)
    lazy_outputs = graph(*inputs)
    lazy_outputs = [output.cpu().numpy() for output in lazy_outputs]
    with open('./outs/bert.json', 'w') as f:
        hidet.utils.netron.dump(graph, f)
    np.testing.assert_allclose(actual=lazy_outputs[0], desired=hidet_outputs[0], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(actual=lazy_outputs[1], desired=hidet_outputs[1], rtol=0.0, atol=0.0)

    # hidet lazy opt
    inputs = [input_ids, attention_mask, token_type_ids]
    symbol_inputs = [hi.symbol_like(t) for t in inputs]
    symbol_outputs = hidet_model(*symbol_inputs)
    graph = hi.trace_from(symbol_outputs, inputs=symbol_inputs)
    with tos.PassContext(
            instruments=[tos.transforms.ProfileInstrument(print_stdout=False)],
            verbose=False
    ):
        graph = hi.tos.optimize(graph)
    for t in range(10):
        cuda_api.device_synchronization()
        with Timer('hidet bert optimized'):
            lazy_outputs = graph(*inputs)
            cuda_api.device_synchronization()
    lazy_outputs = [output.cpu().numpy() for output in lazy_outputs]
    with open('./outs/bert_opt.json', 'w') as f:
        hidet.utils.netron.dump(graph, f)

    # tensor rt
    from hidet.utils.tensorrt_utils import create_engine_from_onnx, engine_inference
    inputs = {
        'input_ids': hidet.array(input_ids).cuda(),
        'attention_mask': hidet.array(attention_mask).cuda(),
        'token_type_ids': hidet.array(token_type_ids).cuda()
    }
    engine = create_engine_from_onnx(model_path, inputs_shape={key: tensor.shape for key, tensor in inputs.items()})
    trt_outputs = engine_inference(engine, inputs)
    trt_outputs = [trt_outputs['last_hidden_state'].cpu().numpy(), trt_outputs['pooler_output'].cpu().numpy()]

    np.testing.assert_allclose(actual=lazy_outputs[0], desired=hidet_outputs[0], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(actual=lazy_outputs[1], desired=hidet_outputs[1], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(actual=trt_outputs[0], desired=hidet_outputs[0], rtol=1e-2, atol=1e-2)
    np.testing.assert_allclose(actual=trt_outputs[1], desired=hidet_outputs[1], rtol=1e-2, atol=1e-2)


if __name__ == '__main__':
    # demo_old_resnet50()
    # demo_resnet50()
    demo_bert()
