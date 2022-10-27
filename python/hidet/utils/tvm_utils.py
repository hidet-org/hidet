import os
from hashlib import sha256
from typing import Optional, Dict, List
import onnx
import tvm
import tvm.relay.backend.executor_factory
from tvm import relay
from tvm import autotvm
from tvm.contrib import graph_executor
from tvm.contrib.graph_executor import GraphModule
import hidet
from hidet import Tensor
from hidet.utils import Timer, hidet_cache_dir
from hidet.testing import benchmark_func


def dump_code(graph_factory: tvm.relay.backend.executor_factory.ExecutorFactoryModule, out_dir):
    runtime_module: tvm.runtime.Module = graph_factory.get_lib()
    runtime_cuda_module = runtime_module.imported_modules[0]
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'tvm_host.cpp'), 'w') as f:
        f.write(runtime_module.get_source())
    with open(os.path.join(out_dir, 'tvm_cuda.cu'), 'w') as f:
        f.write(runtime_cuda_module.get_source())


def dump_relay_cuda_code(ir_module, params=None, out_dir: str = './outs', opt_level=3):
    with tvm.transform.PassContext(opt_level=opt_level):
        graph_module = tvm.relay.build(
            ir_module, target=tvm.target.cuda(arch='sm_60'), target_host=tvm.target.Target('c'), params=params
        )
    # graph_module = tvm.relay.build(ir_module, target='cuda')
    dump_code(graph_module, out_dir)


def autotvm_tune(
    ir_module: tvm.ir.IRModule,
    params: Dict[str, tvm.nd.NDArray],
    target: tvm.target.Target,
    out_dir: str,
    tuner_name='ga',
    num_trial=1000,
) -> None:
    lib_path = os.path.join(out_dir, 'lib.so')

    log_file = os.path.join(out_dir, 'records.json')
    if not os.path.exists(log_file):
        tasks: List[autotvm.task.Task] = autotvm.task.extract_from_program(ir_module, params, target=target)
        with open(os.path.join(out_dir, 'tasks.txt'), 'w') as f:
            for task_idx, task in enumerate(tasks):
                f.write('task {}\n{}\n\n'.format(task_idx, task))

        temp_log_file = log_file + '.tmp'
        with open(temp_log_file, 'a'):
            pass  # create an empty log in case no tunable operators
        with Timer(msg='AutoTVM tuning of {} tasks'.format(len(tasks)), file=os.path.join(out_dir, 'tuning_time.txt')):
            for task_idx, task in enumerate(tasks):
                if tuner_name == 'xgb':
                    tuner = autotvm.tuner.XGBTuner(task)
                elif tuner_name == 'ga':
                    tuner = autotvm.tuner.GATuner(task)
                else:
                    raise ValueError(tuner_name)
                num_trial = min(num_trial, len(task.config_space))
                tuner.tune(
                    n_trial=num_trial,
                    measure_option=autotvm.measure_option(
                        builder=autotvm.LocalBuilder(timeout=10),
                        runner=autotvm.LocalRunner(number=20, repeat=3, timeout=4, min_repeat_ms=150),
                    ),
                    callbacks=[
                        autotvm.callback.progress_bar(num_trial, f'[Task {task_idx:>2}/{len(tasks):<2}]'),
                        autotvm.callback.log_to_file(temp_log_file),
                    ],
                )
        autotvm.record.pick_best(temp_log_file, log_file)
        # os.remove(temp_log_file)

    with autotvm.apply_history_best(log_file):
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(ir_module, target=target, params=params)
    lib.export_library(lib_path)
    dump_code(lib, out_dir)


def ansor_tune(
    ir_module: tvm.ir.IRModule,
    params: Dict[str, tvm.nd.NDArray],
    target: tvm.target.Target,
    out_dir: str,
    num_trial_per_task=800,
):
    from tvm import auto_scheduler

    log_file = os.path.join(out_dir, 'records.json')
    lib_path = os.path.join(out_dir, 'lib.so')

    pair = auto_scheduler.extract_tasks(ir_module, params, target)
    tasks: List[auto_scheduler.SearchTask] = pair[0]
    task_weights: List[int] = pair[1]

    if not os.path.exists(log_file):
        with open(os.path.join(out_dir, 'tasks.txt'), 'w') as f:
            for task_idx, task in enumerate(tasks):
                f.write('task {} (key {})\n{}\n\n'.format(task_idx, task.workload_key, task.compute_dag))
        temp_log_file = log_file + '.temp'
        tune_option = auto_scheduler.TuningOptions(
            num_measure_trials=num_trial_per_task * len(tasks),
            measure_callbacks=[auto_scheduler.RecordToFile(temp_log_file)],
        )
        tuner = auto_scheduler.TaskScheduler(
            tasks,
            task_weights,
            callbacks=[
                auto_scheduler.task_scheduler.PrintTableInfo(),
                auto_scheduler.task_scheduler.LogEstimatedLatency(os.path.join(out_dir, 'estimated_latency.csv')),
            ],
        )
        with Timer(msg='Ansor tuning of {} tasks'.format(len(tasks)), file=os.path.join(out_dir, 'tuning_time.txt')):
            tuner.tune(tune_option)
        os.rename(temp_log_file, log_file)
    with auto_scheduler.ApplyHistoryBest(log_file):
        with tvm.transform.PassContext(opt_level=3, config={'relay.backend.use_auto_scheduler': True}):
            lib = relay.build(ir_module, target, params=params)
    lib.export_library(lib_path)
    dump_code(lib, out_dir)


def build_ir_module(
    ir_module: tvm.ir.IRModule, params: Dict[str, tvm.nd.NDArray], target: tvm.target.Target, out_dir: str
):
    lib_path = os.path.join(out_dir, 'lib.so')
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(ir_module, target, params=params)
    lib.export_library(lib_path)
    dump_code(lib, out_dir)


def tvm_graph_module_from_onnx(
    onnx_model_path: str,
    input_shapes: Optional[Dict[str, List[int]]],
    tune_autotvm=False,
    tune_ansor=False,
    tune_trial_per_task=800,
) -> GraphModule:
    # determine output dir
    if tune_autotvm and tune_ansor:
        raise ValueError('Can not tune network with ansor and autotvm at the same time.')
    if tune_autotvm:
        tuner_name = 'autotvm'
    elif tune_ansor:
        tuner_name = 'ansor'
    else:
        tuner_name = 'notune'
    model_name = os.path.basename(onnx_model_path).rsplit('.', 1)[0]
    cache_dir = hidet_cache_dir(category='tvm_cache')
    hash_key = onnx_model_path + str(input_shapes) + str(tune_trial_per_task)
    out_dir = os.path.join(cache_dir, f'{model_name}_{tuner_name}_{sha256(hash_key.encode()).hexdigest()[:6]}')
    os.makedirs(out_dir, exist_ok=True)

    lib_path = os.path.join(out_dir, 'lib.so')
    if not os.path.exists(lib_path):
        onnx_model = onnx.load_model(onnx_model_path)
        ir_module, params = relay.frontend.from_onnx(onnx_model, input_shapes, dtype='float32')
        target = tvm.target.cuda(arch='sm_{}{}'.format(*hidet.utils.cuda.query_compute_capability()))
        with open(os.path.join(out_dir, 'relay_model.txt'), 'w') as f:
            f.write(str(ir_module))
        with open(os.path.join(out_dir, 'model_info.txt'), 'w') as f:
            lines = [
                'model: {}'.format(onnx_model_path),
                'inputs: {}'.format(str(input_shapes)),
                'ansor: {}'.format(tune_ansor),
                'autotvm: {}'.format(tune_autotvm),
                'trial per task: {}'.format(tune_trial_per_task),
            ]
            f.write('\n'.join(lines))
        if tune_autotvm:
            autotvm_tune(ir_module, params, target, out_dir=out_dir, num_trial=tune_trial_per_task)
        elif tune_ansor:
            ansor_tune(ir_module, params, target, out_dir=out_dir, num_trial_per_task=tune_trial_per_task)
        else:
            build_ir_module(ir_module, params, target, out_dir=out_dir)
        assert os.path.exists(lib_path), 'Failed to generate lib for model {}.'.format(onnx_model_path)
    lib = tvm.runtime.load_module(lib_path)
    device = tvm.cuda()
    gmod = graph_executor.GraphModule(lib['default'](device))
    return gmod


def tvm_inference(gmod: GraphModule, inputs: Dict[str, Tensor]) -> List[Tensor]:
    # currently, TVM does not support get output by name, thus return a list of outputs
    for name, tensor in inputs.items():
        gmod.set_input(name, value=tvm.nd.array(tensor.cpu().numpy()))
    gmod.run()
    outputs = []
    for i in range(gmod.get_num_outputs()):
        output: tvm.nd.NDArray = gmod.get_output(i)
        outputs.append(hidet.array(output.numpy()).cuda())
    return outputs


def tvm_benchmark(gmod: GraphModule, dummy_inputs: Dict[str, Tensor], warmup=10, number=10, repeat=10) -> List[float]:
    for name, tensor in dummy_inputs.items():
        gmod.set_input(name, value=tvm.nd.array(tensor.cpu().numpy()))
    return benchmark_func(lambda: gmod.run(), warmup=warmup, number=number, repeat=repeat, median=False)


def tvm_commit() -> str:
    info = tvm.support.libinfo()
    return info['GIT_COMMIT_HASH']
