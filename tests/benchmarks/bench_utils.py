# Class to initialise backend, run compilation
class Backend:
    def __init__(self, backend, dtype) -> None:
        assert (
            backend == 'hidet' or backend == 'max-autotune' or backend == 'max-autotune-no-cudagraphs'
        ), 'backend is hidet or max-autotune or max-autotune-no-cudagraphs supported only'
        self.backend = backend
        self.dtype = dtype
        if self.backend == 'hidet':
            self.init_hidet()

    def init_hidet(self):
        import hidet, os

        use_fp16 = self.dtype == 'float16'
        hidet.torch.dynamo_config.search_space(2)
        hidet.torch.dynamo_config.use_fp16(use_fp16)
        hidet.torch.dynamo_config.use_fp16_reduction(use_fp16)
        hidet.torch.dynamo_config.use_attention(True)
        hidet.torch.dynamo_config.use_tensor_core(True)
        hidet.torch.dynamo_config.use_cuda_graph(True)
        hidet.option.search_space(2)
        hidet.option.cache_dir(hidet.option.get_cache_dir() + '/regression')

        # hidet.option.parallel_tune(max_parallel_jobs=1)
        # hidet.option.debug_cache_tuning(True)
        # hidet.option.save_lower_ir(True)
        # hidet.option.debug_show_verbose_flow_graph(True)

        # Initialise compiler server
        if os.environ.get('CI_CS_HOSTNAME'):
            hidet.option.compile_server.addr(os.environ.get('CI_CS_HOSTNAME'))
            hidet.option.compile_server.port(int(os.environ.get('CI_CS_PORT')))
            hidet.option.compile_server.username(os.environ.get('CI_CS_USERNAME'))
            hidet.option.compile_server.password(os.environ.get('CI_CS_PASSWORD'))
            hidet.option.compile_server.repo(os.environ.get('REPO_NAME').strip(), os.environ.get('REPO_BRANCH').strip())
            hidet.option.compile_server.enable(flag=True)

    def compile(self, model):
        import torch

        if self.backend == 'hidet':
            model = torch.compile(model, backend=self.backend)
        else:
            model = torch.compile(model, mode=self.backend)
        return model


# Make benchmarking of given torch model
def bench_torch_model(model, torch_inputs, bench_iters=100, warmup_iters=10):
    import torch

    for _ in range(warmup_iters):
        out = model(*torch_inputs)
    torch.cuda.empty_cache()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(bench_iters):
        out = model(*torch_inputs)
    end.record()
    end.synchronize()
    torch.cuda.empty_cache()

    latency = start.elapsed_time(end) / bench_iters
    return latency
