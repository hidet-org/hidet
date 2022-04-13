import os
import numpy as np
import hidet
from hidet import tos
from hidet.utils import hidet_cache_dir


def demo_bert():
    model_name = 'bert-base-uncased'
    model = tos.frontend.from_onnx(os.path.join(hidet_cache_dir(), 'onnx/{}.onnx'.format(model_name)))
    print(model.input_names)
    print(model.output_names)
    batch_size = 1
    seq_length = 128
    vocab_size = 30522
    input_ids = np.random.randint(0, vocab_size, [batch_size, seq_length], dtype=np.int64)
    attention_mask = np.ones(shape=[batch_size, seq_length], dtype=np.int64)
    token_type_ids = np.zeros(shape=[batch_size, seq_length], dtype=np.int64)
    input_ids = hidet.array(input_ids).cuda()
    attention_mask = hidet.array(attention_mask).cuda()
    token_type_ids = hidet.array(token_type_ids).cuda()
    last_hidden_state, pooler_output = model(input_ids, attention_mask, token_type_ids)


if __name__ == '__main__':
    demo_bert()
