# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import re
import os
import json
from hashlib import sha256
from typing import List, Optional, Tuple

import hidet.cuda
from hidet import option
from hidet.ir.stmt import AssertStmt
from hidet.ir.expr import is_constant
from hidet.ir.module import IRModule
from hidet.ir.task import Task
from hidet.drivers.build_module import build_ir_module, build_ir_module_batch
from hidet.drivers.utils import lazy_initialize_cuda
from hidet.runtime.compiled_module import compiled_module_exists
from hidet.runtime.compiled_task import CompiledTask, TensorSignature, load_compiled_task, compiled_task_cache
from hidet.runtime.device import Device
from hidet.utils.multiprocess import parallel_imap
from hidet.utils.py import cyan, green

logger = logging.Logger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def build_task_module(task: Task, candidates: List[IRModule], task_dir: str, target: str):
    from hidet.lang import int32, void
    from hidet.lang import attrs
    from hidet.lang import meta

    # declare the functions to retrieve input/output shapes
    @hidet.script
    def get_input_shape(idx: int32, dims: ~int32):
        attrs.func_kind = 'public'

        for i in meta.range(len(task.inputs)):
            if idx == i:
                for j in meta.range(len(task.inputs[i].shape)):
                    dims[j] = task.inputs[i].shape[j]

    @hidet.script
    def get_output_shape(idx: int32, dims: ~int32):
        attrs.func_kind = 'public'

        for i in meta.range(len(task.outputs)):
            if idx == i:
                for j in meta.range(len(task.outputs[i].shape)):
                    dims[j] = task.outputs[i].shape[j]

    if len(candidates) == 0:
        raise ValueError('No candidate found.')
    elif len(candidates) == 1:
        from hidet.transforms.generate_launch_func import generate_launch_func

        # when there is only one candidate, we reuse the candidate's ir module
        task_ir_module: IRModule = generate_launch_func(candidates[0])
        task_ir_module.add_function(get_input_shape.name, get_input_shape)
        task_ir_module.add_function(get_output_shape.name, get_output_shape)

        launch_func = task_ir_module.functions['launch']
        del task_ir_module.functions['launch']
        if 'launch' in task_ir_module.global_vars:
            del task_ir_module.global_vars['launch']

        launch_func.name = 'launch_0'
        task_ir_module.functions['launch_0'] = launch_func

        object_files = []
    else:
        # otherwise, build each candidate to a .o file, and link them into the task's ir module
        for i, candidate in enumerate(candidates):
            candidate.namespace = f'candidate_{i}'

        # build each candidate to an object file (.o)
        build_ir_module_batch(
            ir_modules=candidates,
            output_dirs=[os.path.join(task_dir, 'candidates', str(i)) for i in range(len(candidates))],
            output_kind='.o',
            target=target,
        )

        param_types = [~t.type.dtype for t in task.params]

        with hidet.script_module() as script_module:

            launch_candidates = []
            for i in range(len(candidates)):
                launch_candidates.append(
                    script_module.declare_extern_func(
                        name='candidate_{}.launch'.format(i), param_types=param_types, ret_type=void
                    )
                )

            for i in range(len(candidates)):

                @hidet.script
                def launch(arg: meta.types(param_types)):
                    attrs.func_name = 'launch_{}'.format(i)
                    attrs.func_kind = 'public'

                    launch_candidates[i](*arg)

        ir_module = script_module.ir_module()
        ir_module.add_function(get_input_shape.name, get_input_shape)
        ir_module.add_function(get_output_shape.name, get_output_shape)
        task_ir_module = ir_module
        object_files = [os.path.join(task_dir, 'candidates', str(i), 'lib.o') for i in range(len(candidates))]

    # add assertions to the launch function
    if len(task.assertions) > 0:
        assertions = tuple(AssertStmt(cond, msg) for cond, msg in task.assertions)
        for _, func in task_ir_module.functions.items():
            # TODO: this is a hacky way of detecting if a function is a launch function
            #    maybe designate a separate function type?
            if func.kind == 'public' and re.search("launch_\d+", func.name):
                body = func.body
                # this should be fine, since ResolveSymbolPass resolves all symbols to the front
                #    before these
                if isinstance(body, hidet.ir.stmt.SeqStmt):
                    body.seq = assertions + body.seq

    # build task ir module
    build_ir_module(
        ir_module=task_ir_module, output_dir=task_dir, output_kind='.so', object_files=object_files, target=target
    )


def generate_meta_data(task: Task, task_dir: str, build_target: str, num_candidates: int):
    from hidet.ir.compute import TensorNode
    from hidet.runtime.compiled_task import TaskMetaData
    from hidet.graph.ops.transfer import TransferTask
    from hidet.utils.dataclass import asdict

    # determine the output device
    if isinstance(task, TransferTask):
        # for transfer tasks, we use the src/dst device to know input/output device
        input_device = str(task.src_device)
        output_device = str(task.dst_device)
    else:
        # for ALL other tasks, their input/output device MUST be the same: the build target
        input_device = output_device = build_target

    def get_signature(t: TensorNode, device: str) -> TensorSignature:
        return TensorSignature(
            device=device, dtype=t.type.dtype.name, shape=[int(v) if is_constant(v) else str(v) for v in t.shape]
        )

    # generate meta data
    meta = TaskMetaData(
        symbols=[v.name for v in task.symbols],
        inputs=[get_signature(t, input_device) for t in task.inputs],
        outputs=[get_signature(t, output_device) for t in task.outputs],
        target=build_target,
        num_candidates=num_candidates,
        hidet_version=hidet.__version__,
    )
    with open(os.path.join(task_dir, 'meta.json'), 'w') as f:
        json.dump(asdict(meta), f, indent=2)


def build_task(task: Task, target='cuda', load=True) -> Optional[CompiledTask]:
    """
    Build a task into a compiled function.

    Parameters
    ----------
    task: Task
        The task to be built.
    target: str
        The target platform. Candidates are 'cuda' and 'cpu'.
    load: bool
        Whether to load the compiled function. If False, the compiled function will not be loaded, and None is returned.
        Otherwise, the compiled function is loaded and returned.
    Returns
    -------
    ret: CompiledTask
        When load is True, the compiled function is returned. Otherwise, None is returned.
    """
    task_string: str = str(task)
    compiled_task: Optional[CompiledTask] = None

    if isinstance(target, Device):
        target = target.kind

    space_level = option.get_option('search_space')
    op_cache_dir = os.path.join(option.get_option('cache_dir'), './ops')
    use_cache = option.get_option('cache_operator')

    # check in-memory cache
    if compiled_task_cache.contains(target, space_level, task_string):
        if load:
            compiled_task = compiled_task_cache.get(target, space_level, task_string)
    else:
        # check on-disk cache
        config_str = f'{target}_space_{space_level}'
        task_hash = sha256(task_string.encode()).hexdigest()[:16]
        task_dir = os.path.join(op_cache_dir, config_str, task.name, task_hash)
        lib_path = os.path.join(task_dir, 'lib.so')
        version_path = os.path.join(task_dir, 'version.txt')

        version_matched = False
        if os.path.exists(version_path):
            with open(version_path, 'r') as f:
                version = f.read()
                if version.strip() == hidet.__version__:
                    version_matched = True

        # use previously generated library when available
        if use_cache and version_matched and compiled_module_exists(task_dir):
            logger.debug(f"Load cached task binary {green(task.name)} from path: \n{cyan(lib_path)}")
            if load:
                compiled_task = load_compiled_task(task_dir)
                compiled_task_cache.add(target, space_level, task_string, compiled_task)
        else:
            logger.info(f"Compiling {target} task {green(task.signature())}...")

            # build from scratch
            os.makedirs(task_dir, exist_ok=True)

            # write task
            with open(os.path.join(task_dir, 'task.txt'), 'w') as f:
                f.write(task_string)

            # write version
            with open(version_path, 'w') as f:
                f.write(hidet.__version__)

            # implement task to IRModule, each task may produce multiple IRModules (candidates)
            # they have the same functionality but different performance
            candidates = task.implement(target=target, working_dir=task_dir)

            # generate meta data
            generate_meta_data(task, task_dir, target, len(candidates))

            # construct the ir module for the task
            build_task_module(task, candidates, task_dir, target)

            if load:
                compiled_task = load_compiled_task(task_dir)
                compiled_task_cache.add(target, space_level, task_string, compiled_task)

    return compiled_task


def build_task_batch(task_target_pairs: List[Tuple[Task, str]]):
    jobs = [(task, target) for task, target in task_target_pairs]

    def build_job(args):
        try:
            task, target = args
            build_task(task, target, load=False)
            return True, 'Success'
        except (Exception,):  # pylint: disable=broad-except
            import traceback

            if option.get_option('parallel_build'):
                return False, traceback.format_exc()
            else:
                raise

    if option.get_option('parallel_build') and len(jobs) > 1:
        lazy_initialize_cuda()
        status_list = list(parallel_imap(build_job, jobs))
    else:
        status_list = list(map(build_job, jobs))
    if not all(status for status, msg in status_list) and option.get_option('parallel_build'):
        msg = ['Failed to build {} tasks:'.format(sum(1 for s, msg in status_list if not s))]
        for (task, target), (status, job_msg) in zip(task_target_pairs, status_list):
            if not status:
                job_msg = ('\n' + job_msg).replace('\n', '\n    ')
                msg.append(f'  [{target}] {task.signature()}')
                msg.append(f'{job_msg}')
        # msg.append('Please turn off parallel build to see the error message:')
        # msg.append('  hidet.option.parallel_build(False)')
        raise RuntimeError('\n'.join(msg))
