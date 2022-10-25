from typing import Tuple, Dict

try:
    import torch
    import torchvision
    from torch import nn
except ImportError:
    pass


def get_torch_model(name: str, batch_size: int = 1, **kwargs) -> Tuple[nn.Module, Dict[str, torch.Tensor]]:
    import transformers
    if name == 'resnet50':
        model = torchvision.models.resnet50(pretrained=True).eval().cuda()
        inputs = {
            'x': torch.randn([batch_size, 3, 224, 224]).cuda()
        }
        return model, inputs
    elif name == 'inception_v3':
        model = torchvision.models.inception_v3(pretrained=True).eval().cuda()
        model.eval()
        inputs = {
            'x': torch.randn([batch_size, 3, 299, 299]).cuda()
        }
        return model, inputs
    elif name == 'mobilenet_v2':
        model = torchvision.models.mobilenet_v2(pretrained=True).eval().cuda()
        inputs = {
            'x': torch.randn([batch_size, 3, 224, 224]).cuda()
        }
        return model, inputs
    elif name == 'bert':
        config = transformers.BertConfig()
        model = transformers.BertModel(config).eval().cuda()
        model.eval()
        vocab_size = 30522
        seq_length = kwargs.get('seq_length', 128)
        inputs = {
            'input_ids': torch.randint(0, vocab_size - 1, size=[batch_size, seq_length]).cuda(),
            'attention_mask': torch.ones(size=[batch_size, seq_length], dtype=torch.int64).cuda(),
            'token_type_ids': torch.zeros(size=[batch_size, seq_length], dtype=torch.int64).cuda()
        }
        return model, inputs
    elif name == 'gpt2':
        config = transformers.GPT2Config()
        model = transformers.GPT2Model(config).eval().cuda()
        model.eval()
        vocab_size = 50257
        seq_length = kwargs.get('seq_length', 128)
        inputs = {
            'input_ids': torch.randint(0, vocab_size - 1, size=[batch_size, seq_length]).cuda(),
            'attention_mask': torch.ones(size=[batch_size, seq_length], dtype=torch.int64).cuda(),
        }
        return model, inputs
    else:
        raise ValueError('Can not recognize model: {}'.format(name))
