import hidet
from hidet.runtime import CompiledFunction
from hidet.ir.dialects.compute import tensor_input
from hidet.graph.ops.schedules.cuda.depthwise_conv import schedule_depthwise_conv2d, Conv2dTask


def dwc_kernel(batch_size, channels, height, width, stride, kernel) -> CompiledFunction:
    in_height, in_width = (height - 1) * stride + kernel, (width - 1) * stride + kernel
    task = Conv2dTask(
        data=tensor_input('x', 'float32', [batch_size, channels, in_height, in_width], 'global'),
        weight=tensor_input('w', 'float32', [channels, 1, kernel, kernel], 'global'),
        stride=[stride, stride],
        groups=channels
    )
    ir_module = schedule_depthwise_conv2d(task)
    func = hidet.driver.build_ir_module(ir_module, func_name='conv2d')
    return func
