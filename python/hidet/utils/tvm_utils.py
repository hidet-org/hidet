import os


def dump_relay_cuda_code(ir_module, out_dir: str = './outs'):
    import tvm.relay
    import tvm.target
    graph_module = tvm.relay.build(ir_module, target='cuda', target_host=tvm.target.Target('c'))
    # graph_module = tvm.relay.build(ir_module, target='cuda')
    runtime_module: tvm.runtime.Module = graph_module.get_lib()
    runtime_cuda_module = runtime_module.imported_modules[0]
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'tvm_host.cpp'), 'w') as f:
        f.write(runtime_module.get_source())
    with open(os.path.join(out_dir, 'tvm_cuda.cu'), 'w') as f:
        f.write(runtime_cuda_module.get_source())

