from hidet import tos


def demo_bert():
    model_name = 'bert-base-uncased'
    model = tos.frontend.from_onnx('~/model_zoo/{}.onnx'.format(model_name))


if __name__ == '__main__':
    demo_bert()
