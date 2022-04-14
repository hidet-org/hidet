import pytest
import numpy as np
import os

import hidet.utils
from hidet import randn
from hidet.tos.frontend.onnx_utils import OnnxModule
from hidet.utils import download


def onnx_installed() -> bool:
    try:
        import onnx
        import onnxruntime
    except ImportError:
        return False


def transformers_installed() -> bool:
    try:
        import transformers
        import transformers.onnx
    except ImportError:
        return False


@pytest.mark.skipif(condition=onnx_installed(), reason='ONNX not installed')
def test_resnet50():
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


@pytest.mark.skipif(condition=onnx_installed(), reason='Package onnx or onnxruntime not installed.')
@pytest.mark.skipif(condition=transformers_installed(), reason='Package transformers or transformers[onnx] not installed.')
def test_bert(batch_size=1, seq_length=128):
    import onnx
    import onnxruntime
    model_cache_dir = hidet.utils.hidet_cache_dir(category='onnx')
    os.makedirs(model_cache_dir, exist_ok=True)
    model_path = hidet.utils.transformers_utils.export_transformer_model_as_onnx(
        model_name='bert-base-uncased',
        feature='default',
        output_dir=model_cache_dir
    )
    # inputs: input_ids, attention_mask, token_type_ids
    # outputs: last_hidden_state, pooler_output
    onnx.checker.check_model(model_path)

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

    # hidet
    hidet_model = hidet.tos.frontend.from_onnx(model_path)
    input_ids = hidet.array(input_ids).cuda()
    attention_mask = hidet.array(attention_mask).cuda()
    token_type_ids = hidet.array(token_type_ids).cuda()
    hidet_outputs = hidet_model(input_ids, attention_mask, token_type_ids)
    hidet_outputs = [out.cpu().numpy() for out in hidet_outputs]

    np.testing.assert_allclose(actual=hidet_outputs[0], desired=onnx_outputs[0], rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(actual=hidet_outputs[1], desired=onnx_outputs[1], rtol=1e-5, atol=1e-5)


if __name__ == '__main__':
    pytest.main([__name__])
