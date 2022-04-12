import os
from hidet import tos
from hidet.utils import hidet_cache_dir


def demo_bert():
    model_name = 'bert-base-uncased'
    model = tos.frontend.from_onnx(os.path.join(hidet_cache_dir(), 'onnx/{}.onnx'.format(model_name)))


if __name__ == '__main__':
    demo_bert()
