import numpy as np
import os

import tvm
from tvm import relay, autotvm
from tvm.relay import testing
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.autotvm.graph_tuner import DPTuner, PBQPTuner
import tvm.contrib.graph_executor as runtime

x = relay.Var("x", tvm.relay.TensorType([512, 512]))
y = relay.Var("y", tvm.relay.TensorType([512, 512]))

params = {}

# mod = relay.Function(
#     [x, y],
#     relay.nn.dense(x, y)
# )

# mod = tvm.IRModule.from_expr(mod)
net = relay.nn.dense(x, y)
mod = relay.Function(relay.analysis.free_vars(net), net)

target = "llvm -mcpu=core-avx2"
# target = "c"

batch_size = 1
dtype = "float32"
model_name = "dense_model_1"
log_file = "logs-%s.log" % model_name
graph_opt_sch_file = "logs-%s_graph_opt.log" % model_name

# input_name = "data"

num_threads = 32
os.environ["TVM_NUM_THREADS"] = str(num_threads)


tuning_option = {
    "log_filename": log_file,
    "tuner": "xgb",
    "early_stopping": None,
    "measure_option": autotvm.measure_option(
        builder=autotvm.LocalBuilder(),
        runner=autotvm.LocalRunner(
            number=1, repeat=10, min_repeat_ms=0, enable_cpu_cache_flush=True, timeout=100
        )
    )
}


def tune_kernels(
    tasks, measure_option, tuner="gridsearch", early_stopping=None, log_filename=log_file
):
    for i, task in enumerate(tasks):
        prefix = "[Task %2d / %2d] " % (i + 1, len(tasks))

        # create tuner
        if tuner == "xgb":
            tuner_obj = XGBTuner(task, loss_type="reg")
        elif tuner == "xgb_knob":
            tuner_obj = XGBTuner(task, loss_type="reg", feature_type="knob")
        elif tuner == "xgb_itervar":
            tuner_obj = XGBTuner(task, loss_type="reg", feature_type="itervar")
        elif tuner == "xgb_curve":
            tuner_obj = XGBTuner(task, loss_type="reg", feature_type="curve")
        elif tuner == "xgb_rank":
            tuner_obj = XGBTuner(task, loss_type="rank")
        elif tuner == "xgb_rank_knob":
            tuner_obj = XGBTuner(task, loss_type="rank", feature_type="knob")
        elif tuner == "xgb_rank_itervar":
            tuner_obj = XGBTuner(task, loss_type="rank", feature_type="itervar")
        elif tuner == "xgb_rank_curve":
            tuner_obj = XGBTuner(task, loss_type="rank", feature_type="curve")
        elif tuner == "xgb_rank_binary":
            tuner_obj = XGBTuner(task, loss_type="rank-binary")
        elif tuner == "xgb_rank_binary_knob":
            tuner_obj = XGBTuner(task, loss_type="rank-binary", feature_type="knob")
        elif tuner == "xgb_rank_binary_itervar":
            tuner_obj = XGBTuner(task, loss_type="rank-binary", feature_type="itervar")
        elif tuner == "xgb_rank_binary_curve":
            tuner_obj = XGBTuner(task, loss_type="rank-binary", feature_type="curve")
        elif tuner == "ga":
            tuner_obj = GATuner(task, pop_size=50)
        elif tuner == "random":
            tuner_obj = RandomTuner(task)
        elif tuner == "gridsearch":
            tuner_obj = GridSearchTuner(task)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        # do tuning
        max_ntrials = 750
        n_trial = min(len(task.config_space), max_ntrials)
        # n_trial = 1500
        tuner_obj.tune(
            n_trial=n_trial,
            early_stopping=early_stopping,
            measure_option=measure_option,
            callbacks=[
                autotvm.callback.progress_bar(n_trial, prefix=prefix),
                autotvm.callback.log_to_file(log_filename),
            ],
        )


# Use graph tuner to achieve graph level optimal schedules
# Set use_DP=False if this takes too long to finish.
def tune_graph(graph, dshape, records, opt_sch_file, use_DP=True):
    target_op = [
        relay.op.get("nn.dense")
    ]
    Tuner = DPTuner if use_DP else PBQPTuner
    executor = Tuner(graph, {"x": dshape, "y": dshape}, records, target_op, target)
    executor.benchmark_layout_transform(min_exec_num=2000)
    executor.run()
    executor.write_opt_sch2record_file(opt_sch_file)


# Launch tuning jobs and evaluate the end-to-end performance
def evaluate_performance(lib, data_shape):
    # upload parameters to device
    dev = tvm.cpu()
    x_tvm = tvm.nd.array((np.random.randn(*data_shape).astype(dtype)))
    y_tvm = tvm.nd.array((np.random.randn(*data_shape).astype(dtype)))
    module = runtime.GraphModule(lib["default"](dev))
    module.set_input("x", x_tvm)
    module.set_input("y", y_tvm)

    # evaluate
    print("Evaluate inference time cost...")
    print(module.benchmark(dev, number=20, repeat=3))


def tune_and_evaluate(tuning_opt):
    print("Extract tasks...")
    tasks = autotvm.task.extract_from_program(
        mod, target=target, params=params, ops=(relay.op.get("nn.dense"), )
    )

    data_shape = [512, 512]

    # run tuning tasks
    tune_kernels(tasks, **tuning_opt)
    # tune_graph(mod, data_shape, log_file, graph_opt_sch_file)

    # compile kernels in kernel tuned only mode
    print("\nEvaluation of the network been tuned on kernel level: ")
    with autotvm.apply_history_best(log_file):
        print("Compile...")
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, target=target, params=params)
        evaluate_performance(lib, data_shape)


tune_and_evaluate(tuning_option)



















