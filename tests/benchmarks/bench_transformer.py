import os
import argparse
import torch

from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM, AutoModelForCausalLM
from transformers import AutoModelForQuestionAnswering, logging
from hidet.testing.torch_utils import bench_torch_model, bench_gen_model, Backend

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.set_verbosity_error()

model_class = {
    'bert-base-uncased': 'AutoModelForMaskedLM',
    'meta-llama/Llama-2-7b-hf': 'AutoModelForCausalLM',
    'meta-llama/Llama-2-7b-chat-hf': 'AutoModelForCausalLM',
    'meta-llama/CodeLlama-7b-hf': 'AutoModelForCausalLM',
    'google/gemma-2b': 'AutoModelForCausalLM',
    'mistralai/Mistral-7B-v0.1': 'AutoModelForCausalLM',
    'openai-community/gpt2-xl': 'AutoModelForCausalLM',
    'mosaicml/mpt-7b': 'AutoModelForCausalLM',
    'DiscoResearch/mixtral-7b-8expert': 'AutoModelForCausalLM',
    # 'mistralai/Mixtral-8x7B-v0.1': 'AutoModelForCausalLM',
}


def get_full_model_name(model_name):
    short_to_full_model_name = {
        'bert-base-uncased': 'bert-base-uncased',
        'llama-2-7b': 'meta-llama/Llama-2-7b-chat-hf',
        'gemma-2b': 'google/gemma-2b',
        'mistral': 'mistralai/Mistral-7B-v0.1',
        'gpt2-xl': 'openai-community/gpt2-xl',
        'mpt-7b': 'mosaicml/mpt-7b',
        'codellama-7b': 'meta-llama/CodeLlama-7b-hf',
        # 'mixtral': 'mistralai/Mixtral-8x7B-v0.1',
        'mixtral': 'DiscoResearch/mixtral-7b-8expert',
    }
    return short_to_full_model_name[model_name]


def bench_causal_lm(model_name, bs, genlen, dtype, backend, mode):
    comp_backend = Backend(backend, mode, dtype)

    dtype = getattr(torch, dtype)
    model_name = get_full_model_name(model_name)
    AutoModel_cls = eval(model_class[model_name])

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel_cls.from_pretrained(model_name)
    model = model.eval().to(dtype).cuda()

    # Input
    INPUT_STRING = """
    I'm going to the interview to startup CentML om position of Software Engeneer. 
    Want to prepare to the interview. Could you describe in details what does CentML do?
    """

    input_string_batch = [INPUT_STRING] * bs
    inputs = tokenizer(input_string_batch, return_tensors='pt')['input_ids'].cuda()

    with torch.no_grad(), torch.autocast("cuda"):
        model = comp_backend.compile(model)
        latency = bench_gen_model(model, tokenizer, inputs, bs=bs, genlen=genlen)
        del model
    return latency


def bench_masked_lm(model_name, seqlen, bs, dtype, backend, mode):
    comp_backend = Backend(backend, mode, dtype)
    dtype = getattr(torch, dtype)
    model_name = get_full_model_name(model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    AutoModel_cls = eval(model_class[model_name])
    model = AutoModel_cls.from_pretrained(model_name, max_position_embeddings=8192, ignore_mismatched_sizes=True)
    model = model.eval().to(dtype).cuda()
    # Input
    INPUT_STRING = "The capital of France is [MASK]. The capital of England is [MASK]. The of Australia is [MASK].The capital of Canada is [MASK]."
    input_string_batch = [INPUT_STRING] * bs
    seqlen = min(seqlen, tokenizer.model_max_length)
    inputs = tokenizer(
        input_string_batch, padding='max_length', truncation=True, max_length=seqlen, return_tensors='pt'
    )['input_ids'].cuda()

    with torch.no_grad(), torch.autocast("cuda"):
        model = comp_backend.compile(model)
        latency = bench_torch_model(model, [inputs])
        del model
    return latency


if __name__ == '__main__':
    SEQLEN_DEFAULT = 256
    BS_DEFAULT = 1
    GENLEN_DEFAULT = 1

    parser = argparse.ArgumentParser(prog='Benchmark Transformers')
    parser.add_argument('model', type=str, help='Specify model')
    parser.add_argument(
        '--params',
        type=str,
        default=f'seqlen={SEQLEN_DEFAULT}',
        help='Specify Input Parameters. E.g., [bs=1][,][seqlen=256][,][genlen=1]',
    )
    parser.add_argument('--dtype', type=str, default='float16', help='Specify precision. E.g., float32')
    parser.add_argument('--backend', type=str, default='hidet', help='torch.compile backend')
    parser.add_argument('--mode', type=str, default='max-autotune', help='torch.compile mode')
    args = parser.parse_args()

    model_name, dtype, backend, mode = args.model, args.dtype, args.backend, args.mode

    seqlen = SEQLEN_DEFAULT
    bs = BS_DEFAULT
    genlen = GENLEN_DEFAULT
    params = args.params.split(',')
    for p in params:
        name, value = p.split('=')
        if name == 'bs':
            bs = int(value)
        elif name == 'seqlen':
            seqlen = int(value)
        elif name == 'genlen':
            genlen = int(value)

    if model_class[get_full_model_name(model_name)] == 'AutoModelForMaskedLM':
        latency = bench_masked_lm(model_name, seqlen, bs, dtype, backend, mode)
    elif model_class[get_full_model_name(model_name)] == 'AutoModelForCausalLM':
        latency = bench_causal_lm(model_name, bs=bs, genlen=genlen, dtype=dtype, backend=backend, mode=mode)

    print(latency)
