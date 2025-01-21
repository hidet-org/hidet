import os
import argparse
import numpy as np
import torch

from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM, AutoModelForCausalLM
from transformers import AutoModelForQuestionAnswering, logging
from hidet.testing.torch_utils import bench_model, bench_gen_model, Backend

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.set_verbosity_error()

model_class = {
    'bert-base-uncased': 'AutoModelForMaskedLM',
    'meta-llama/Llama-2-7b-hf': 'AutoModelForCausalLM',
    'meta-llama/Llama-2-7b-chat-hf': 'AutoModelForCausalLM',
    'google/gemma-2b': 'AutoModelForCausalLM',
    'mistralai/Mistral-7B-v0.1': 'AutoModelForCausalLM',
    'openai-community/gpt2-xl': 'AutoModelForCausalLM',
    'albert/albert-base-v2': 'AutoModelForMaskedLM',
    'allenai/longformer-base-4096': 'AutoModelForMaskedLM',
    'facebook/bart-base': 'AutoModelForCausalLM',
    'facebook/blenderbot-400M-distill': 'AutoModelForCausalLM',
    'almanach/camembert-base': 'AutoModelForCausalLM',
    'microsoft/deberta-v3-large': 'AutoModelForMaskedLM',
    'distilbert/distilbert-base-uncased': 'AutoModelForMaskedLM',
    'distilbert/distilgpt2': 'AutoModelForCausalLM',
    'google/electra-base-generator': 'AutoModelForCausalLM',
    'EleutherAI/gpt-j-6B': 'AutoModelForCausalLM',
    'EleutherAI/gpt-neo-1.3B': 'AutoModelForCausalLM',
    'google/fnet-base': 'AutoModelForMaskedLM',
    'microsoft/layoutlm-base-uncased': 'AutoModelForMaskedLM',
    'facebook/mbart-large-cc25': 'AutoModelForCausalLM',
    'google/mobilebert-uncased': 'AutoModelForMaskedLM',
    'facebook/opt-350m': 'AutoModelForCausalLM',
    'hf-tiny-model-private/tiny-random-PLBartForCausalLM': 'AutoModelForCausalLM',
    'google/pegasus-large': 'AutoModelForCausalLM',
    'FacebookAI/roberta-base': 'AutoModelForCausalLM',
    'facebook/xglm-564M': 'AutoModelForCausalLM',
    'YituTech/conv-bert-base': 'AutoModelForMaskedLM',
    'mosaicml/mpt-7b': 'AutoModelForCausalLM',
    'codellama/CodeLlama-7b-hf': 'AutoModelForCausalLM',
    'DiscoResearch/mixtral-7b-8expert': 'AutoModelForCausalLM',
    'meta-llama/CodeLlama-7b-hf': 'AutoModelForCausalLM',
}

short_to_full_model_name = {
    'llama-2-7b': 'meta-llama/Llama-2-7b-chat-hf',
    'mixtral-7b-8': 'DiscoResearch/mixtral-7b-8expert',
    'codellama-7b': 'meta-llama/CodeLlama-7b-hf',
    'mpt-7b': 'mosaicml/mpt-7b',
    'bert-base-uncased': 'bert-base-uncased',
    'gemma-2b': 'google/gemma-2b',
    'mistral': 'mistralai/Mistral-7B-v0.1',
    'gpt2-xl': 'openai-community/gpt2-xl',
    'albert-base-v2': 'albert/albert-base-v2',
    'longformer': 'allenai/longformer-base-4096',
    'bart-causal': 'facebook/bart-base',
    'blender-bot-causal': 'facebook/blenderbot-400M-distill',
    'camembert': 'almanach/camembert-base',
    'deberta': 'microsoft/deberta-v3-large',
    'distilbert-base-uncased': 'distilbert/distilbert-base-uncased',
    'distillgpt2': 'distilbert/distilgpt2',
    'electra': 'google/electra-base-generator',
    'gpt-j': 'EleutherAI/gpt-j-6B',
    'gpt-neo': 'EleutherAI/gpt-neo-1.3B',
    'google-fnet': 'google/fnet-base',
    'layoutlm': 'microsoft/layoutlm-base-uncased',
    'mbart': 'facebook/mbart-large-cc25',
    'mobilebert-uncased': 'google/mobilebert-uncased',
    'opt': 'facebook/opt-350m',
    'plt-bast': 'hf-tiny-model-private/tiny-random-PLBartForCausalLM',
    'pegasus': 'google/pegasus-large',
    'roberta': 'FacebookAI/roberta-base',
    'xglm': 'facebook/xglm-564M',
    'conv-bert': 'YituTech/conv-bert-base',
}


def get_full_model_name(model_name):
    return short_to_full_model_name[model_name]


def bench_causal_lm(model_name, bs, genlen, dtype, backend, mode, cache):
    comp_backend = Backend(backend, mode, dtype, cache)

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

    input_string_batch = INPUT_STRING * bs
    inputs = tokenizer(input_string_batch, return_tensors='pt')['input_ids'].cuda()
    END_OF_SENTENCE_ID = tokenizer.eos_token_id

    with torch.no_grad(), torch.autocast("cuda"):
        _, torch_output = bench_gen_model(model, tokenizer, inputs, genlen=genlen)
        # Temporary workaround for gpt-j
        # gpt-j initializes tensors during the first forwasd pass
        # which causes recompilation during the second forward pass
        if model_name == 'EleutherAI/gpt-j-6B':
            model(inputs)
        model = comp_backend.compile(model)
        latency, hidet_output = bench_gen_model(model, tokenizer, inputs, genlen=genlen)
        del model

    torch.testing.assert_close(hidet_output, torch_output, rtol=0, atol=0)
    return latency


def bench_masked_lm(model_name, seqlen, bs, dtype, backend, mode, cache):
    comp_backend = Backend(backend, mode, dtype, cache)
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
        torch_output = model(inputs)
        model = comp_backend.compile(model)
        latency = bench_model(model, [inputs], true_outputs=torch_output)
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
    parser.add_argument('--cache', type=str, default='', help='')

    args = parser.parse_args()

    model_name, dtype, backend, mode, cache = args.model, args.dtype, args.backend, args.mode, args.cache

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
        latency = bench_masked_lm(model_name, seqlen, bs, dtype, backend, mode, cache)
    elif model_class[get_full_model_name(model_name)] == 'AutoModelForCausalLM':
        latency = bench_causal_lm(
            model_name, bs=bs, genlen=genlen, dtype=dtype, backend=backend, mode=mode, cache=cache
        )

    print(latency)
