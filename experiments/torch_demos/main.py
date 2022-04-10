import torch
from torchvision.models.resnet import resnet50
from hidet.utils import Timer
from hidet.ffi.cuda_api import CudaAPI
import transformers
from transformers import pipeline
from transformers import BertTokenizer


def demo_resnet50():
    model = resnet50().cuda()
    model.train(False)
    x = torch.rand(16, 3, 224, 224).cuda()
    for t in range(10):
        CudaAPI.device_synchronization()
        with Timer(f'torch {t}'):
            y = model(x)
            CudaAPI.device_synchronization()
        y = None


def demo_transformer():
    # classifier = pipeline('sentiment-analysis')
    # print(classifier('We are very happy to show you the 🤗 Transformers library.'))
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    text = 'Hello, world!'
    print(tokenizer.tokenize(text))
    print(tokenizer(text))
    encoded_seq = tokenizer(text)['input_ids']
    print(tokenizer.decode(encoded_seq))


def demo_bert():
    tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
    model = transformers.BertForSequenceClassification.from_pretrained('bert-base-uncased')
    inputs = tokenizer(['Hello, world!', 'hhh'], return_tensors='pt', padding=True)

    print(inputs)
    outputs = model(**inputs)
    print(outputs.logits)


def demo_bert_pt2onnx(model_name='bert-base-uncased', batch_size=8, seq_length=128):
    tokenizer = transformers.BertTokenizer.from_pretrained(model_name)
    model = transformers.BertForSequenceClassification.from_pretrained(model_name)
    inputs = tokenizer(['' for _ in range(batch_size)], return_tensors='pt', padding='max_length', max_length=seq_length)

    print(inputs)
    outputs = model(**inputs)
    print(outputs.logits)
    torch.onnx.export(model, inputs, './outs/{}_bs{}_seq_{}.onnx'.format(model_name, batch_size, seq_length))


if __name__ == '__main__':
    # demo_resnet50()
    # demo_transformer()
    # demo_bert()
    demo_bert_pt2onnx()
