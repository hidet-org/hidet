from hidet.cli.bench.model import BenchModel


class NLPModel(BenchModel):
    def __init__(self, repo_name, model_name, label, batch_size: int, sequence_length: int):
        self.repo_name = repo_name
        self.model_name = model_name
        self.label = label
        self.batch_size = batch_size
        self.sequence_length = sequence_length

    def __str__(self):
        return '{}/{}'.format(self.model_name, self.label)

    def model(self):
        import torch

        return torch.hub.load(self.repo_name, self.model_name, self.label)

    def example_inputs(self):
        import torch

        tokens_tensor = torch.zeros((self.batch_size, self.sequence_length), dtype=torch.long, device='cuda')
        segments_tensors = torch.zeros((self.batch_size, self.sequence_length), dtype=torch.long, device='cuda')
        args = (tokens_tensor,)
        kwargs = {'token_type_ids': segments_tensors}
        return args, kwargs
