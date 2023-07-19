import os
import json
import numpy as np
import torch
import torchvision
import hidet
import argparse
from result_entry import ResultEntry, ResultGroup, load_regression_data
from transformers import AutoTokenizer, AutoModelForMaskedLM, logging
from torch import _dynamo
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.set_verbosity_error()

device_name = str(hidet.cuda.properties().name, 'UTF-8')

def bench_torch_model(model, torch_inputs, bench_iters=10, warmup_iters=5):
    for _ in range(warmup_iters):
        torch_out = model(*torch_inputs)
    torch.cuda.empty_cache()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(bench_iters):
        torch_out = model(*torch_inputs)
    end.record()
    end.synchronize()
    torch.cuda.empty_cache()

    latency = start.elapsed_time(end) / bench_iters
    return latency

def bench_hf_transformers(model_name, seqlen, dtype):
    use_fp16 = dtype == 'float16'
    hidet.torch.dynamo_config.search_space(2)
    hidet.torch.dynamo_config.use_fp16(use_fp16)
    hidet.torch.dynamo_config.use_fp16_reduction(use_fp16)
    hidet.torch.dynamo_config.use_attention(True)
    hidet.torch.dynamo_config.use_tensor_core(True)
    dtype = getattr(torch, dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name,
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

def bench_torchvision(model_cls, shape, dtype):
    use_fp16 = dtype == 'float16'
    hidet.torch.dynamo_config.search_space(2)
    hidet.torch.dynamo_config.use_fp16(use_fp16)
    hidet.torch.dynamo_config.use_fp16_reduction(use_fp16)
    hidet.torch.dynamo_config.use_attention(True)
    hidet.torch.dynamo_config.use_tensor_core(True)
    dtype = getattr(torch, dtype)
    model = model_cls(weights=None)
    model = model.eval().to(dtype).cuda()
    torch_inputs = [torch.randn(shape, device='cuda', dtype=dtype)]
    with torch.no_grad(), torch.autocast("cuda"):
        model = torch.compile(model, backend='hidet')
        latency = bench_torch_model(model, torch_inputs)
        del model
    return latency

def bert_regression():
    regression_data = load_regression_data()
    result_group = ResultGroup(name='bert-base Regression', device_name=device_name)
    bert_data = regression_data[device_name]['bert_base_shapes']
    for shape, perf in bert_data.items():
        for dtype, ref_latency in perf.items():
            [seqlen] = [int(s.strip()) for s in shape.split(',')]
            latency = bench_hf_transformers('bert-base-uncased', seqlen, dtype)
            if not np.allclose(latency, ref_latency, rtol=0.05):
                # ToDo: deal with slowdown/speedup
                pass
            result_group.add_entry(ResultEntry(shape, dtype, latency))
    return result_group

def resnet_regression():
    regression_data = load_regression_data()
    result_group = ResultGroup(name='resnet50 Regression', device_name=device_name)
    resnet50_data = regression_data[device_name]['resnet50_shapes']
    model_cls = torchvision.models.resnet50
    for shape, perf in resnet50_data.items():
        for dtype, ref_latency in perf.items():
            _shape = [int(s.strip()) for s in shape.split(',')]
            latency = bench_torchvision(model_cls, _shape, dtype)
            if not np.allclose(latency, ref_latency, rtol=0.05):
                # ToDo: deal with slowdown/speedup
                pass
            result_group.add_entry(ResultEntry(shape, dtype, latency))
    return result_group


def efficientnet_regression():
    # ToDo
    return None

def llama_regression():
    # ToDo
    return None

def model_performance_regression(report_file):
    result_groups = []
    result_groups.append(bert_regression())
    result_groups.append(resnet_regression())
    result_groups.append(efficientnet_regression())
    result_groups.append(llama_regression())
    with open(report_file, 'w') as f:
        f.write("---------------- Model Performance Regression -----------------\n")
        for result_group in result_groups:
            if result_group is not None:
                f.write(str(result_group))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Model Performance Regression')
    parser.add_argument(
        '--report',
        type=str,
        default='./report_model_performance.txt',
        help='Specify report output path'
    )
    args = parser.parse_args()
    model_performance_regression(args.report)