from typing import List

import click

import hidet
from tqdm import tqdm
import torch
from hidet import FlowGraph
from gpt_model import gpt2
from encoder import get_encoder

hidet.option.search_space(0)

bucket_size = 50


class GPT2Generator:
    def __init__(self, max_num_tokens, model_size, use_fp16):
        import hidet.cuda.graph

        graph: FlowGraph = gpt2(seq_length=max_num_tokens, model_size=model_size, use_fp16=use_fp16)
        self.cuda_graph: hidet.cuda.graph.CudaGraph = graph.cuda_graph()
        self.encoder = get_encoder()
        self.max_num_tokens = max_num_tokens

        # get the torch view for the two hidet tensors
        self.input_ids = self.cuda_graph.inputs[0].torch()
        self.logits = self.cuda_graph.outputs[0].torch()

    def __call__(self, text: str) -> str:
        ids: List[int] = self.encoder.encode(text)
        num_init_tokens = len(ids)
        if num_init_tokens > self.max_num_tokens:
            return text

        self.input_ids[:num_init_tokens] = torch.asarray(ids)

        for i in tqdm(range(num_init_tokens, self.max_num_tokens), "generating", ncols=80):
            self.cuda_graph.run()
            next_token: int = torch.argmax(self.logits[i - 1], dim=-1).item()
            self.input_ids[i] = next_token

        output_ids = self.input_ids[num_init_tokens:].cpu().tolist()
        output_text = self.encoder.decode(output_ids)
        return output_text


@click.command()
@click.option("--max-num-tokens", default=40, type=int, help='Max number of total tokens to process and generate',
              show_default=True)
@click.option("--use-fp16", is_flag=True, default=False, help='Use fp16', show_default=True)
@click.option("--model-size", default="124M", type=click.Choice(['124M', '355M', '774M', '1558M']), show_default=True)
@click.option("--tune", is_flag=True, default=False,
              help='Tune the operators for better performance. May take several minutes.', show_default=True)
def main(max_num_tokens: int, use_fp16: bool, model_size: str, tune: bool):
    if tune:
        hidet.option.search_space(2)
    generator = GPT2Generator(max_num_tokens, model_size, use_fp16)
    while True:
        x = click.prompt(">>> ", type=str, prompt_suffix="")
        response = generator(x)
        click.echo(x + response)


if __name__ == "__main__":
    main()
