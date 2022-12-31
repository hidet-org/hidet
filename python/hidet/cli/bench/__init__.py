from hidet.graph.frontend.torch import availability as torch_availability
from .bench import bench_group

if not torch_availability.dynamo_available():
    raise RuntimeError(
        'PyTorch version is less than 2.0. Please upgrade PyTorch to 2.0 or higher to enable torch dynamo'
        'which is required by the benchmark scripts.'
    )
