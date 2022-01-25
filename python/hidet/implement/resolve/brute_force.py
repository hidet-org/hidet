import random
import numpy as np
from itertools import product
from hidet.ir.func import IRModule, FunctionGroup
from hidet.runtime.value import dummy_inputs_from_task
from hidet.backend import build
from tqdm import tqdm


def brute_force_resolve(ir_module: IRModule, warmup=1, number=1, repeat=10, progress_bar=True) -> IRModule:
    task_name = ir_module.task.name
    max_trials = int(1e6)

    groups = []
    for name in ir_module.functions:
        if not isinstance(ir_module.functions[name], FunctionGroup):
            continue
        group: FunctionGroup = ir_module.functions[name]
        groups.append(group)

    num_trials = 1
    for g in groups:
        num_trials *= len(g.group)
    assert num_trials <= max_trials, f"num_trails {num_trials} exceeds the maximum {max_trials}"

    group_index = {group: idx for idx, group in enumerate(groups)}

    candidates = []
    candidates_label = []

    for indices in product(*[range(len(g.group)) for g in groups]):
        candidate = IRModule()
        for name, func in ir_module.functions.items():
            if isinstance(func, FunctionGroup):
                candidate.add(name, func.group[indices[group_index[func]]])
            else:
                candidate.add(name, func)
        candidates.append(candidate)
        label = "|".join(f'{group.name}:{group.group[indices[idx]].get_attr("label")}' for idx, group in enumerate(groups))
        candidates_label.append(label)

    candidates_latency = []
    inputs = dummy_inputs_from_task(ir_module.task)
    pbar = tqdm(zip(candidates, candidates_label), total=num_trials, disable=not progress_bar)
    best_latency = 1e9
    name_limit = 64
    for candidate, label in pbar:
        if len(label) > name_limit:
            label = label[:name_limit] + str(hash(name_limit))
        module = build(candidate, f'./outs/resolve/{label}', keep=False)
        latencies = module[task_name].profile(*inputs, warmup=warmup, number=number, repeat=repeat)
        current_latency = float(np.mean(latencies))
        best_latency = min(best_latency, current_latency)
        candidates_latency.append(current_latency)
        pbar.set_description('Best: {:.3f} ms Current: {:.3f} ms'.format(best_latency, current_latency))

    return candidates[np.argmin(candidates_latency)]
