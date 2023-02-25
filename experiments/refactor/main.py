import numpy.testing

import hidet
import torch
import torch.backends.cuda
import torch.backends.cudnn
from hidet.ir.tools import astext

# hidet.option.parallel_build(False)
hidet.option.save_lower_ir()
hidet.option.cache_dir('./outs/cache')


def debug_simplier():
    from hidet.ir.primitives.cuda import threadIdx, blockIdx
    from hidet.transforms.rule_based_simplifier import RuleBasedSimplifier
    simplifier = RuleBasedSimplifier()
    e = (((blockIdx.x * 500) + threadIdx.x) / 1)
    s = simplifier(e)
    print(e)
    print(s)


def demo_e2e():
    a = hidet.symbol([3, 4], device='cuda')
    b = hidet.symbol([4, 5], device='cuda')
    c = hidet.ops.matmul(a, b)
    task: hidet.ir.Task = c.op.task
    print(astext(task))
    ir_module = task.implement('cuda', './outs/working')
    print(astext(ir_module))
    hidet.driver.build_task(
        task,
        target_device='cuda'
    )
    # hidet.driver.build_ir_module(
    #     ir_module=ir_module,
    #     func_name='matmul',
    #     output_dir='./outs/work',
    #     save_ir=True,
    #     profile_pass=True,
    #     load=False
    # )
    # print(c.op.latency())

def demo_end_to_end():
    a = hidet.randn([3, 4], device='cuda')
    b = hidet.randn([4, 5], device='cuda')
    c = hidet.ops.matmul(a, b)


def demo_maxpool():
    a = hidet.randn([1, 1, 1, 1], device='cuda')
    # aa = a.torch()
    # bb = torch.nn.functional.max_pool2d(aa, kernel_size=2, stride=1, padding=1)
    b = hidet.ops.max_pool2d(a, kernel=3, stride=1, padding=1)
    print(a)
    print(b)
    # numpy.testing.assert_allclose(b.cpu().numpy(), bb.cpu().numpy(), atol=1e-5, rtol=1e-5)


def demo_resnet():
    model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True).eval().cuda()
    x = torch.randn(1, 3, 224, 224).cuda()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    hidet.torch.dynamo_config.correctness_report()

    model_opt = torch.compile(model, backend='hidet')
    y = model_opt(x)

def demo_bool():
    a = hidet.ir.Expr()
    if a and False:
        print(a)

def demo_bert():
    from hidet.testing.onnx_models import get_onnx_model
    model_path, input_names, input_tensors = get_onnx_model('bert', batch_size=1)
    model = hidet.graph.frontend.from_onnx(model_path)
    input_tensors = [t.cuda() for t in input_tensors]
    graph = model.flow_graph_for(input_tensors)

def demo_gather():
    hidet.option.cache_operator(False)
    a = hidet.asarray([1, 128, 768]).cuda()
    b = hidet.asarray(2).cuda()
    c = hidet.ops.take(a, b, axis=0).cuda()
    print(a)
    print(b)
    print(c)

def demo_functor():
    from hidet.ir.task import Task
    from hidet.ir.tools import rewrite
    a = hidet.asarray([1, 128, 768]).cuda()
    b = hidet.asarray(2).cuda()
    # c = hidet.ops.take(a, b, axis=0).cuda()
    c = hidet.ops.take(hidet.symbol_like(a), b, axis=0).cuda()
    task: Task = c.op.task
    node = task.outputs[0]
    out = rewrite(node, {})
    print(node is out)


def main():
    # debug_simplier()
    # demo_e2e()
    # demo_end_to_end()
    # demo_resnet()
    # demo_maxpool()
    # test_bool()
    demo_bert()
    # demo_gather()
    # demo_functor()


if __name__ == '__main__':
    main()

