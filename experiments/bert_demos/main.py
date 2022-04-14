import os
import numpy as np
import hidet
from hidet import tos
from hidet.utils import hidet_cache_dir


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
    x_hidet = randn([1, 3, 224, 224], device='cuda')
    module = OnnxModule(model)
    y_hidet = module(x_hidet)
    y_hidet = y_hidet.cpu().numpy()

    # run use onnx runtime
    onnx_infer = onnxruntime.InferenceSession(model_path)
    y_onnx = onnx_infer.run(None, {'data': x_hidet.cpu().numpy()})
    y_onnx = y_onnx[0]

    # compare
    np.testing.assert_allclose(actual=y_hidet, desired=y_onnx, rtol=1e-5, atol=1e-5)

def demo_bert():
    import onnx
    import onnxruntime
    model_name = 'bert-base-uncased'
    model_path = os.path.join(hidet_cache_dir(), 'onnx/{}.onnx'.format(model_name))

    batch_size = 1
    seq_length = 128
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


if __name__ == '__main__':
    demo_resnet50()
    # demo_bert()
