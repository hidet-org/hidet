from typing import Tuple, List
import numpy as np
import hidet
from hidet.tos import Tensor
from hidet.utils import download, hidet_cache_dir, hidet_cache_file
from hidet.utils.transformers_utils import export_transformer_model_as_onnx
from hidet.utils.torch_utils import export_torchvision_model_as_onnx
from .model_blocks import get_bert_block, get_resnet50_block
from .operators import get_onnx_operator


def get_onnx_model(name: str, batch_size: int = 1, **kwargs) -> Tuple[str, List[str], List[Tensor]]:
    """
    kwargs candidates:
      seq_length=128
    """
    if name == 'resnet50':
        model_path = hidet_cache_file('onnx', f'{name}.onnx')
        export_torchvision_model_as_onnx(model_name=name, output_path=model_path)
        input_names = ['data']
        input_tensors = [hidet.randn(shape=[batch_size, 3, 224, 224])]
        return model_path, input_names, input_tensors
    elif name == 'inception_v3':
        model_path = hidet_cache_file('onnx', f'{name}.onnx')
        export_torchvision_model_as_onnx(model_name=name, output_path=model_path)
        input_names = ['data']
        input_tensors = [hidet.randn(shape=[batch_size, 3, 299, 299])]
        return model_path, input_names, input_tensors
    elif name == 'mobilenet_v2':
        model_path = hidet_cache_file('onnx', f'{name}.onnx')
        export_torchvision_model_as_onnx(model_name=name, output_path=model_path)
        input_names = ['data']
        input_tensors = [hidet.randn(shape=[batch_size, 3, 224, 224])]
        return model_path, input_names, input_tensors
    elif name == 'bert':
        model_path = hidet_cache_file('onnx', 'bert.onnx')
        export_transformer_model_as_onnx(
            model_name='bert-base-uncased',
            output_path=model_path
        )
        vocab_size = 30522
        seq_length = kwargs.get('seq_length', 128)
        input_names = [
            'input_ids',
            'attention_mask',
            'token_type_ids'
        ]
        input_tensors = [
            hidet.array(np.random.randint(0, vocab_size-1, size=[batch_size, seq_length], dtype=np.int64)),
            hidet.ones(shape=[batch_size, seq_length], dtype='int64'),
            hidet.zeros(shape=[batch_size, seq_length], dtype='int64')
        ]
        return model_path, input_names, input_tensors
    elif name == 'bart':
        model_path = hidet_cache_file('onnx', 'bart.onnx')
        export_transformer_model_as_onnx(
            model_name='facebook/bart-base',
            output_path=model_path
        )
        vocab_size = 50265
        seq_length = kwargs.get('seq_length', 128)
        input_names = [
            'input_ids',
            'attention_mask',
            'decoder_input_ids',
            'decoder_attention_mask'
        ]
        input_tensors = [
            hidet.array(np.random.randint(0, vocab_size-1, size=[batch_size, seq_length], dtype=np.int64)),
            hidet.ones(shape=[batch_size, seq_length], dtype='int64'),
            hidet.array(np.random.randint(0, vocab_size-1, size=[batch_size, seq_length], dtype=np.int64)),
            hidet.ones(shape=[batch_size, seq_length], dtype='int64')
        ]
        return model_path, input_names, input_tensors
    elif name == 'gpt2':
        model_path = hidet_cache_file('onnx', 'gpt2.onnx')
        export_transformer_model_as_onnx(
            model_name='gpt2',
            output_path=model_path
        )
        vocab_size = 50257
        seq_length = kwargs.get('seq_length', 128)
        input_names = [
            'input_ids',
            'attention_mask',
        ]
        input_tensors = [
            hidet.array(np.random.randint(0, vocab_size-1, size=[batch_size, seq_length], dtype=np.int64)),
            hidet.ones(shape=[batch_size, seq_length], dtype='int64'),
        ]
        return model_path, input_names, input_tensors
    elif name.startswith('resnet50_'):
        return get_resnet50_block(name, batch_size=batch_size, **kwargs)
    elif name.startswith('bert_'):
        return get_bert_block(name, batch_size=batch_size, **kwargs)
    elif name.startswith('op_'):
        return get_onnx_operator(name, batch_size)
    else:
        raise NotImplementedError('Can not recognize model {}'.format(name))


if __name__ == '__main__':
    names = [
        'resnet50',
        'inception_v3',
        'mobilenet_v2',
        'bert',
        'bart',
        'gpt2'
    ]
    configs = {
        'bert': {'seq_length': 512},
        'bart': {'seq_length': 512},
        'gpt2': {'seq_length': 512},
    }
    for model_name in names:
        kwargs = {}
        if model_name in configs:
            kwargs.update(configs[model_name])
        get_onnx_model(model_name, **kwargs)
