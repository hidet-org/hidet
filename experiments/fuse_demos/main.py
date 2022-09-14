from typing import Union, List, Tuple
import hidet
from hidet import ops
from hidet import Tensor, FlowGraph
from hidet.ir.func import IRModule
from hidet.graph.transforms.fuse_operator import FuseOperatorPass
from hidet.graph import nn


def main():
    import hidet
    from hidet import ops

    @hidet.jit(opt=True, parallel_k='disabled')
    def func(a: Tensor, b: Tensor, c: Tensor, d: Tensor):
        e = a + b
        f = ops.matmul(e, c)
        g = f + d
        return g

    a = hidet.randn([1, 4, 4])
    b = hidet.randn([1, 4, 4])
    c = hidet.randn([1, 4, 4])
    d = hidet.randn([1, 4, 4])

    graph: FlowGraph = func.flow_graph_for(a, b, c, d)
    print(graph)

    fused_graph = FuseOperatorPass()(graph)
    print(fused_graph)


def demo_2():
    import hidet
    from hidet import ops

    @hidet.jit(opt=False, parallel_k='disabled')
    def func(a: Tensor):
        b = a + a
        c = ops.matmul(b, a)
        d = c + b
        return d

    a = hidet.randn([1, 4, 4])

    graph: FlowGraph = func.flow_graph_for(a)
    print(graph)

    fused_graph = FuseOperatorPass()(graph)
    print(fused_graph)

    for op in fused_graph.nodes:
        print(op.name, op.task)


def demo_resnet():
    import hidet
    import hidet.testing
    module: hidet.graph.Module = hidet.testing.tos_models.resnet.resnet50()

    graph: FlowGraph = module.flow_graph_for([hidet.randn([1, 3, 224, 224])])

    fused_graph = FuseOperatorPass()(graph)

    with open('./fused_resnet50.json', 'w') as f:
        hidet.utils.netron.dump(fused_graph, f)

    print(fused_graph)


def demo_bert():
    import hidet.testing
    module, inputs = hidet.testing.tos_models.bert.bert()
    graph = module.flow_graph_for(inputs)

    with open('./origin_bert.json', 'w') as f:
        hidet.utils.netron.dump(graph, f)

    with hidet.utils.Timer(msg='fuse pass'):
        fused_graph = FuseOperatorPass()(graph)

    with open('./fused_bert.json', 'w') as f:
        hidet.utils.netron.dump(fused_graph, f)


def demo_attention():
    import hidet.testing
    from hidet.testing.tos_models.bert import BertConfig, BertSelfAttention, BertLayer
    batch_size = 1
    seq_length = 128
    bert_config = BertConfig()
    # attention = BertSelfAttention(bert_config)
    module = BertLayer(bert_config)
    hidden_states = hidet.randn([batch_size, seq_length, bert_config.hidden_size])
    attention_mask = hidet.ones([batch_size, seq_length], dtype='int64')

    graph = module.flow_graph_for([hidden_states, attention_mask])

    with open('./origin_bert_layer.json', 'w') as f:
        hidet.utils.netron.dump(graph, f)

    fused_graph = FuseOperatorPass()(graph)

    with open('./fused_bert_layer.json', 'w') as f:
        hidet.utils.netron.dump(fused_graph, f)


def demo_bert_output():
    import hidet.testing
    from hidet.testing.tos_models.bert import BertConfig, BertSelfAttention, BertLayer, BertOutput
    batch_size = 1
    seq_length = 128
    bert_config = BertConfig()
    module = BertOutput(bert_config)
    hidden_states = hidet.randn([batch_size, seq_length, bert_config.intermediate_size])
    skip_hidden_states = hidet.randn([batch_size, seq_length, bert_config.hidden_size])

    graph = module.flow_graph_for([hidden_states, skip_hidden_states])

    with open('./origin_bert_output.json', 'w') as f:
        hidet.utils.netron.dump(graph, f)

    fused_graph = FuseOperatorPass()(graph)

    with open('./fused_bert_output.json', 'w') as f:
        hidet.utils.netron.dump(fused_graph, f)


def demo_layer_norm():
    import hidet.testing
    from hidet.testing.tos_models.bert import BertConfig, BertSelfAttention, BertLayer, BertOutput

    module = hidet.graph.nn.LayerNorm([10])

    x = hidet.randn([20, 10])
    graph = module.flow_graph_for([x])

    with open('./origin_layer_norm.json', 'w') as f:
        hidet.utils.netron.dump(graph, f)

    fused_graph = FuseOperatorPass()(graph)

    with open('./fused_layer_norm.json', 'w') as f:
        hidet.utils.netron.dump(fused_graph, f)


from hidet.ir.task import Task
from hidet.ir.dialects.compute import TensorNode, compute
from hidet.graph.ir import Operator
from hidet.graph.ops.definitions.utils import input_like


def demo_apply():
    import hidet.testing
    from hidet.testing.tos_models.bert import BertConfig, BertSelfAttention, BertLayer, BertOutput

    class FusibleModule(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            # return ops.matmul(x + x, x) + x
            # return ops.matmul(x + x, x)
            return ops.reduce_sum(x, dims=[0], keep_dim=True) + x
            # return ops.reduce_sum(x + x, dims=[0], keep_dim=True)

    module = FusibleModule()

    x = hidet.randn([1, 10, 10])
    graph = module.flow_graph_for([x])

    with open('./origin_layer_norm.json', 'w') as f:
        hidet.utils.netron.dump(graph, f)

    fused_graph: FlowGraph = FuseOperatorPass()(graph)
    print(fused_graph)

    node = fused_graph.nodes[0]
    task = node.task
    print(task)

    # from hidet.graph.ops.schedules.cuda.auto_scheduler import CudaAutoScheduler
    # scheduler = CudaAutoScheduler()
    # ir_module: IRModule = scheduler.schedule_task(task, 'cuda')
    ir_module: IRModule = node.task.implement('cuda')
    # print(ir_module)

    func = ir_module.lookup(task.name + '_grid')
    # func = ir_module.lookup('compute_c')
    print(func)

    from hidet.transforms.tools import apply_prologue_epilogue, generate_packed_func
    fused_func = apply_prologue_epilogue(ir_module, func, task)
    generate_packed_func(ir_module, fused_func, task.name)
    print(ir_module)


def demo_reduce():
    @hidet.jit()
    def func(x: Tensor):
        return ops.reduce_sum(x, [0])

    x = hidet.ones([10, 10])
    y = func(x)
    print(x)
    print(y)


Ints = Union[int, List[int], Tuple[int]]


class BasicConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Ints, padding: Ints = 0, stride: Ints = 1, groups: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding, stride, groups)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        # x = self.bn(x)
        # return ops.relu(x)
        return x


class InceptionB(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.branch3x3 = BasicConv2d(in_channels, 384, kernel_size=3, stride=2)
        self.branch3x3_p1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3_p2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3_p3 = BasicConv2d(96, 96, kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3(x)

        branch3x3_p = self.branch3x3_p1(x)
        branch3x3_p = self.branch3x3_p2(branch3x3_p)
        branch3x3_p = self.branch3x3_p3(branch3x3_p)

        branch_pool = ops.max_pool2d(x, kernel=3, stride=2, padding=0)

        return ops.concat([branch3x3, branch3x3_p, branch_pool], axis=1)


class InceptionC(nn.Module):
    def __init__(self, in_channels: int, channels_7x7: int):
        super().__init__()
        # self.branch1x1 = BasicConv2d(in_channels, 192, kernel_size=1)

        c7 = channels_7x7
        # self.branch7x7_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        # self.branch7x7_2 = BasicConv2d(c7, c7, kernel_size=[1, 7], padding=[0, 3])
        # self.branch7x7_3 = BasicConv2d(c7, 192, kernel_size=[7, 1], padding=[3, 0])

        self.branch7x7_dbl = BasicConv2d(in_channels, c7, kernel_size=[7, 1], padding=[3, 0])
        self.branch7x7_dbl_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7_dbl_2 = BasicConv2d(c7, c7, kernel_size=[7, 1], padding=[3, 0])
        self.branch7x7_dbl_3 = BasicConv2d(c7, c7, kernel_size=[1, 7], padding=[0, 3])
        self.branch7x7_dbl_4 = BasicConv2d(c7, c7, kernel_size=[7, 1], padding=[3, 0])
        self.branch7x7_dbl_5 = BasicConv2d(c7, 192, kernel_size=[1, 7], padding=[0, 3])

        # self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        # branch7x7_dbl = self.branch7x7_dbl_1(x)
        # branch7x7_dbl = self.branch7x7_dbl_2(branch7x7_dbl)
        # branch7x7_dbl = self.branch7x7_dbl_3(branch7x7_dbl)
        # branch7x7_dbl = self.branch7x7_dbl_4(branch7x7_dbl)
        # branch7x7_dbl = self.branch7x7_dbl_5(branch7x7_dbl)

        # branch7x7_dbl = self.branch7x7_dbl_1(x)
        # branch7x7_dbl = self.branch7x7_dbl_2(branch7x7_dbl)
        # branch7x7_dbl = self.branch7x7_dbl_2(x)
        # branch7x7_dbl = self.branch7x7_dbl_2(x)
        branch7x7_dbl = self.branch7x7_dbl_3(x)
        # branch7x7_dbl = self.branch7x7_dbl_4(branch7x7_dbl)
        # branch7x7_dbl = self.branch7x7_dbl_5(branch7x7_dbl)

        return branch7x7_dbl


def inception_b(batch_size=1):
    inputs = [hidet.randn([batch_size, 288, 35, 35])]
    model = InceptionB(in_channels=288)
    return model, inputs


n = 128


def inception_c(in_channels=n, channels_7x7=n, batch_size=1):
    # assert (in_channels, channels_7x7) in [(128, 128), (768, 128), (768, 160), (768, 160), (768, 192)]
    inputs = [hidet.randn([batch_size, in_channels, 17, 17])]
    model = InceptionC(in_channels, channels_7x7=channels_7x7)
    return model, inputs


def demo_inception():
    import numpy as np
    import numpy.testing
    import hidet.testing
    from hidet.ir.utils import validate_schedule
    # module, inputs = hidet.testing.tos_models.inception.inception_v3()
    # module, inputs = hidet.testing.tos_models.inception.inception_c()
    module, inputs = inception_c()
    graph = module.flow_graph_for(inputs)
    y1 = graph(*inputs)

    with hidet.graph.PassContext() as ctx:
        ctx.save_graph_instrument(out_dir='./outs/{}'.format(module.__class__.__name__))
        graph_opt = hidet.graph.optimize(graph)
        # for node in graph_opt.nodes:
        #     task = node.task
        #     assert validate_schedule(task, 'cuda'), 'validate failed: \n{}'.format(task)
        # for node in graph_opt.nodes:
        #     validate_schedule(node.task, 'cuda')

        y2 = graph_opt(*inputs)

    numpy.testing.assert_allclose(y1.numpy(), y2.numpy(), rtol=1e-2, atol=1e-2)


def demo_debug():
    from hidet.ir.utils import validate_schedule
    import numpy.testing
    # x = hidet.randn([1, 8, 289, 128])
    #
    # @hidet.jit(opt=False)
    # def func(x):
    #     # x: [1, 8, 289, 128]
    #     x = ops.reduce_sum(x, dims=[1], keep_dim=False)
    #     x = ops.reshape(x, [1, 1, 17, 17, 128])
    #     x = ops.rearrange(x, [[0], [4], [2], [3]])
    #     return x

    x = hidet.randn([1, 128, 17, 17])
    # x = hidet.randn([1, 128, 17, 23])
    @hidet.jit(opt=False)
    def func(x):
        x = ops.pad(x, [3, 3])
        x = ops.conv2d_gemm_image_transform(x, [1, 7], stride=[1, 1])
        x = ops.reshape(x, [1, 289, 8, 112])
        x = ops.rearrange(x, [[2], [1], [3]])
        x = ops.batched_matmul(x, hidet.randn([8, 112, 128]))
        x = ops.reshape(x, [1, 8, 289, 128])

        # x = ops.reduce_sum(x, dims=[1], keep_dim=False)
        # x = ops.reshape(x, [1, 1, 17, 17, 128])
        # x = ops.rearrange(x, [[0], [4], [2], [3]])
        return x

    graph = func.flow_graph_for(x)

    y1 = graph(x)
    with hidet.graph.PassContext() as ctx:
        ctx.set_parallel_k(disabled=True)
        ctx.save_graph_instrument(out_dir='./outs/{}'.format('debug'))
        graph_opt = hidet.graph.optimize(graph)
        for node in graph_opt.nodes:
            print(node.task)
        y2 = graph_opt(x)

    numpy.testing.assert_allclose(y1.numpy(), y2.numpy(), rtol=1e-2, atol=1e-2)


if __name__ == '__main__':
    # main()
    # demo_2()
    # demo_resnet()
    # demo_bert()
    # demo_attention()
    # demo_bert_output()
    # demo_layer_norm()
    # demo_apply()
    # demo_reduce()
    # demo_inception()
    demo_debug()