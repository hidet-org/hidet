import time
import argparse
import torch
import resource

from transformers import AutoTokenizer, AutoModelForCausalLM

from hidet.testing.torch_utils import Backend
from bench_transformer import model_class, get_full_model_name


def bench_comptime_causal_lm(model_name, bs, genlen, dtype, backend, mode, cache):
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
        # Temporary workaround for gpt-j
        # gpt-j initializes tensors during the first forwasd pass
        # which causes recompilation during the second forward pass
        if model_name == 'EleutherAI/gpt-j-6B':
            model(inputs)
        model = comp_backend.compile(model)
        start = time.time()
        model(inputs)
        end = time.time()
        comp_time = end - start
        del model

    return comp_time


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

    if model_class[get_full_model_name(model_name)] == 'AutoModelForCausalLM':
        comp_time = bench_comptime_causal_lm(
            model_name, bs=bs, genlen=genlen, dtype=dtype, backend=backend, mode=mode, cache=cache
        )
    else:
        raise ValueError('Model class not supported')

    print(f"Execution time: {comp_time:.2f}")

    # Peak memory for the main process
    peak_memory_self = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    # Peak memory for all child processes
    peak_memory_children = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024
    print(f"Peak Memory (Main Process): {peak_memory_self:.2f} MB")
    print(f"Peak Memory (All Child Processes): {peak_memory_children:.2f} MB")
    print(f"Total Peak Memory: {peak_memory_self + peak_memory_children:.2f} MB")
