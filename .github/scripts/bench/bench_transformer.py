import sys
import os
import argparse
import numpy as np
import torch
import hidet
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM, logging
from bench_utils import enable_compile_server, setup_hidet_flags, bench_torch_model
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.set_verbosity_error()

model_class = {
    'bert-base-uncased': 'AutoModelForMaskedLM',
}

def bench_hf_transformers(model_name, seqlen, dtype):
    setup_hidet_flags(dtype)
    enable_compile_server(True)
    dtype = getattr(torch, dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    AutoModel_cls = eval(model_class[model_name])
    model = AutoModel_cls.from_pretrained(model_name,
            max_position_embeddings=8192, ignore_mismatched_sizes=True)
    model = model.eval().to(dtype).cuda()
    inputs = tokenizer("Dummy sentence", padding='max_length', max_length=seqlen,
                       return_tensors='pt')
    inputs = {'input_ids': inputs['input_ids']}
    torch_inputs = tuple(i.clone().cuda() for i in inputs.values())
    with torch.no_grad(), torch.autocast("cuda"):
        model = torch.compile(model, backend='hidet')
        latency = bench_torch_model(model, torch_inputs)
        del model
    return latency

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Benchmark Transformers')
    parser.add_argument(
        'model',
        type=str,
        help='Specify model'
    )
    parser.add_argument(
        '--params',
        type=str,
        default='seqlen=1024',
        help='Specify Input Parameters. E.g., seqlen=1024'
    )
    parser.add_argument(
        '--dtype',
        type=str,
        default='float16',
        help='Specify precision. E.g., float32'
    )
    args = parser.parse_args()

    model, dtype = args.model, args.dtype
    seqlen = int(args.params.split('=')[1])
    latency = bench_hf_transformers(model, seqlen, dtype)
    print(latency)