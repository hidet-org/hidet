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

def enable_compiled_server():
    # TODO: Accept compile server config as input to regression launching
    # Uncomment and edit below options to use your own compile server. See apps/compile_server README.
    # hidet.option.compile_server.addr('xx.xx.xx.xx')
    # hidet.option.compile_server.port(0)
    # hidet.option.compile_server.username('username')
    # hidet.option.compile_server.password('password')
    # hidet.option.compile_server.repo('https://github.com/hidet-org/hidet', 'main')
    # hidet.option.compile_server.enable()
    pass

def setup_hidet_flags(dtype):
    use_fp16 = dtype == 'float16'
    hidet.torch.dynamo_config.search_space(2)
    hidet.torch.dynamo_config.use_fp16(use_fp16)
    hidet.torch.dynamo_config.use_fp16_reduction(use_fp16)
    hidet.torch.dynamo_config.use_attention(True)
    hidet.torch.dynamo_config.use_tensor_core(True)
    hidet.torch.dynamo_config.use_cuda_graph(True)
    hidet.torch.dynamo_config.dump_graph_ir("./graph_ir")

def bench_torch_model(model, torch_inputs, bench_iters=100, warmup_iters=10):
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
    setup_hidet_flags(dtype)
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
    setup_hidet_flags(dtype)
    dtype = getattr(torch, dtype)
    model = model_cls(weights=None)
    model = model.eval().to(dtype).cuda()
    torch_inputs = [torch.randn(shape, device='cuda', dtype=dtype)]
    with torch.no_grad(), torch.autocast("cuda"):
        model = torch.compile(model, backend='hidet')
        latency = bench_torch_model(model, torch_inputs)
        del model
    return latency

def bench_torchhub(repo_name, model_name, shape, dtype):
    setup_hidet_flags(dtype)
    dtype = getattr(torch, dtype)
    model = torch.hub.load(repo_name, model_name, pretrained=True)
    model = model.eval().to(dtype).cuda()
    torch_inputs = [torch.randn(shape, device='cuda', dtype=dtype)]
    with torch.no_grad(), torch.autocast("cuda"):
        model = torch.compile(model, backend='hidet')
        latency = bench_torch_model(model, torch_inputs)
        del model
    return latency

def bert_regression():
    print("Running regression for bert-base")
    regression_data = load_regression_data()
    result_group = ResultGroup(name='bert-base Regression', device_name=device_name)
    bert_data = regression_data[device_name]['bert_base_shapes']
    for shape, perf in bert_data.items():
        for dtype, ref_latency in perf.items():
            [seqlen] = [int(s.strip()) for s in shape.split(',')]
            latency = bench_hf_transformers('bert-base-uncased', seqlen, dtype)
            result_group.add_entry(ResultEntry(shape, dtype, latency, ref_latency))
    return result_group

def torchvision_regression(model_name):
    print("Running regression for", model_name)
    regression_data = load_regression_data()
    result_group = ResultGroup(name=model_name + ' Regression', device_name=device_name)
    perf_data = regression_data[device_name][model_name + '_shapes']
    if any(name in model_name for name in ['deeplab', 'fcn', 'lraspp']):
        model_cls = getattr(torchvision.models.segmentation, model_name)
    else:
        model_cls = getattr(torchvision.models, model_name)
    for shape, perf in perf_data.items():
        for dtype, ref_latency in perf.items():
            if dtype == 'float32':
                continue
            _shape = [int(s.strip()) for s in shape.split(',')]
            latency = bench_torchvision(model_cls, _shape, dtype)
            result_group.add_entry(ResultEntry(shape, dtype, latency, ref_latency))
    return result_group

def torchhub_regression(repo_name, model_name):
    print("Running regression for", model_name)
    regression_data = load_regression_data()
    result_group = ResultGroup(name=model_name + ' Regression', device_name=device_name)
    perf_data = regression_data[device_name][model_name + '_shapes']
    for shape, perf in perf_data.items():
        for dtype, ref_latency in perf.items():
            _shape = [int(s.strip()) for s in shape.split(',')]
            latency = bench_torchhub(repo_name, model_name, _shape, dtype)
            result_group.add_entry(ResultEntry(shape, dtype, latency, ref_latency))
    return result_group

def llama_regression():
    # TODO: Add llama regression
    return None

def model_performance_regression(report_file):
    # Uncomment below line to limit parallel jobs if running out of CPU memory
    # hidet.option.parallel_tune(16)
    hidet.option.cache_dir(hidet.option.get_cache_dir() + '/regression')
    result_groups = []
    result_groups.append(torchvision_regression('resnet50'))
    result_groups.append(torchvision_regression('deeplabv3_resnet50'))
    result_groups.append(torchvision_regression('mobilenet_v2'))
    result_groups.append(torchvision_regression('efficientnet_b0'))
    result_groups.append(torchvision_regression('alexnet'))
    result_groups.append(torchvision_regression('vgg19'))
    result_groups.append(torchvision_regression('squeezenet1_1'))
    result_groups.append(torchvision_regression('inception_v3'))
    result_groups.append(torchvision_regression('googlenet'))
    result_groups.append(torchvision_regression('shufflenet_v2_x1_0'))
    result_groups.append(torchvision_regression('regnet_x_400mf'))
    result_groups.append(torchvision_regression('densenet121'))
    result_groups.append(bert_regression())
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
    enable_compiled_server()
    model_performance_regression(args.report)
