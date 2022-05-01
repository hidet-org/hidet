from typing import List, Tuple
import numpy as np
import os
import onnx
import onnxruntime

import hidet.utils
from hidet import symbol_like, Tensor
from hidet.utils.py import cyan, green, error_tolerance


def check_model(
        model_path: str,
        input_names: List[str],
        input_tensors: List[Tensor],
        mode: str,
        precision: str,
        reduce_precision: str,
        mma: str
):
    onnx.checker.check_model(model_path)

    # onnx
    onnx_session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])  # use cpu executor for high accuracy
    onnx_outputs = onnx_session.run(None, input_feed={name: tensor.numpy() for name, tensor in zip(input_names, input_tensors)})

    # hidet
    hidet_model = hidet.tos.frontend.from_onnx(model_path)
    hidet_inputs = [hidet.array(tensor).cuda() for tensor in input_tensors]

    if mode == 'imperative':
        hidet_outputs = hidet_model(*hidet_inputs)
    elif mode == 'traced' or mode == 'opt':
        symbol_inputs = [symbol_like(tensor) for tensor in hidet_inputs]
        symbol_outputs = hidet_model(*symbol_inputs)
        graph = hidet.trace_from(symbol_outputs, symbol_inputs)
        if mode == 'opt':
            model_name = os.path.splitext(os.path.basename(model_path))[0]
            with hidet.tos.PassContext() as ctx:
                ctx.save_graph_instrument(out_dir=os.path.join('./outs/', model_name, '{}_{}_{}_{}'.format(mode, precision, reduce_precision, mma)))
                ctx.set_precision(precision)
                ctx.set_reduce_precision(reduce_precision)
                ctx.set_mma(mma)
                # ctx.profile_pass_instrument(print_stdout=True)
                graph = hidet.tos.optimize(graph)
        hidet_outputs = graph(*hidet_inputs)
    else:
        raise ValueError()

    if isinstance(hidet_outputs, Tensor):
        hidet_outputs = [hidet_outputs]
    hidet_outputs = [tensor.numpy() for tensor in hidet_outputs]

    assert len(onnx_outputs) == len(hidet_outputs)
    et = 0.0
    for onnx_output, hidet_output in zip(onnx_outputs, hidet_outputs):
        et = max(et, error_tolerance(hidet_output, onnx_output))
        # np.testing.assert_allclose(actual=hidet_output, desired=onnx_output, rtol=1e-4, atol=1e-4)
    return et


def check_all_onnx_models(
        model_name: str,
        batch_size=1,
        mode: str = 'imperative',
        precision: str = 'float32',
        reduce_precision: str = 'float32',
        mma: str = 'simt'
):
    from hidet.testing.onnx_models import get_onnx_model

    assert model_name in ['resnet50', 'inception_v3', 'mobilenet_v2', 'bert', 'bart', 'gpt2']
    assert mode in ['imperative', 'traced', 'opt']

    # print('checking model {} in {} mode'.format(model_name, mode))
    model_path, input_names, input_tensors = get_onnx_model(model_name, batch_size=batch_size)
    et = check_model(model_path, input_names, input_tensors, mode, precision, reduce_precision, mma)
    print('batch_size {:2}  mode {:10}  model {:10}  precision {:8}  reduce_precision {:8}  mma {:10}  error_tolerance {}'.format(
        green(batch_size), green(mode), green(model_name), green(precision), green(reduce_precision), green(mma), green(et, '{:.5f}')
    ))


def main():
    batch_size = 1
    for model_name in [
        'resnet50',
        'inception_v3',
        'mobilenet_v2',
        'bert',
        'gpt2'
    ]:
        for mode in [
            # 'traced',
            # 'imperative',
            'opt'
        ]:
            for precision in [
                'float32',
                'float16',
                'bfloat16'
            ]:
                for reduce_precision in [
                    'float32',
                    *(['float16'] if precision == 'float16' else [])
                ]:
                    for mma in [
                        'simt',
                        'wmma',
                        # 'mma'
                    ]:
                        check_all_onnx_models(
                            model_name=model_name,
                            batch_size=batch_size,
                            mode=mode,
                            precision=precision,
                            reduce_precision=reduce_precision,
                            mma=mma
                        )


if __name__ == '__main__':
    # demo_resnet50_opt()
    # demo_resnet50()
    # demo_bert()
    # hidet.driver.logger.setLevel('DEBUG')
    main()
