# pylint: disable=subprocess-run-check
import os
import pickle
import argparse
import sys
import subprocess
import inspect
import tempfile
import importlib

_ncu_path: str = '/usr/local/cuda/bin/ncu'
_ncu_ui_path: str = '/usr/local/cuda/bin/ncu-ui'
_ncu_ui_template = "{ncu_ui_path} {report_path}"
_ncu_template = """
{ncu_path}
--export {report_path}
--force-overwrite
--set full
--rule CPIStall 
--rule FPInstructions 
--rule HighPipeUtilization 
--rule IssueSlotUtilization 
--rule LaunchConfiguration 
--rule Occupancy 
--rule PCSamplingData 
--rule SOLBottleneck 
--rule SOLFPRoofline 
--rule SharedMemoryConflicts 
--rule SlowPipeLimiter 
--rule ThreadDivergence 
--rule UncoalescedGlobalAccess
--rule UncoalescedSharedAccess 
--import-source yes
--check-exit-code yes
{python_executable} {python_script} {args}
""".replace(
    '\n', ' '
).strip()


class NsightComputeReport:
    def __init__(self, report_path: str):
        self.report_path: str = report_path

    def visualize(self):
        command = _ncu_ui_template.format(ncu_ui_path=_ncu_ui_path, report_path=self.report_path)
        subprocess.run(command, shell=True)


def _ncu_run_func(script_path, func_name, args_pickled_path):
    with open(args_pickled_path, 'rb') as f:
        args, kwargs = pickle.load(f)

    # remove the dir path of the current script from sys.path to avoid module overriding
    sys.path = [path for path in sys.path if not path.startswith(os.path.dirname(__file__))]

    try:
        sys.path.append(os.path.dirname(script_path))
        module = importlib.import_module(os.path.basename(script_path)[:-3])
    except Exception as e:
        raise RuntimeError('Can not import the python script: {}'.format(script_path)) from e

    if not hasattr(module, func_name):
        raise RuntimeError('Can not find the function "{}" in {}'.format(func_name, script_path))

    func = getattr(module, func_name)

    try:
        func(*args, **kwargs)
    except Exception as e:
        raise RuntimeError('Error when running the function "{}"'.format(func_name)) from e


def ncu_set_path(ncu_path: str):
    """
    Set the path to the ncu executable that will be used.

    Parameters
    ----------
    ncu_path: str
        The path to the ncu executable.
    """
    # pylint: disable=global-variable-not-assigned
    global _ncu_path
    _ncu_path = ncu_path


def ncu_get_path():
    """
    Get the path to the ncu executable that will be used.

    Returns
    -------
    str:
        The path to the ncu executable.
    """
    return _ncu_path


def ncu_run(func, *args, **kwargs) -> NsightComputeReport:
    """
    Use nsight compute to profile the call to the given function with (*args, **kwargs) arguments.

    The profile results will be stored in "ncu-reports" subdirectory of the callee function's directory.

    Usage:

    ```python
        from hidet.utils.ncu_utils import ncu_run

        def func():
            ...

        if __name__ == '__main__':
            # we need to wrap this part into '__main__' as this script will be imported in the utility script

            # the function will be profiled and the profiling result will be stored in "ncu-reports" subdirectory.
            report = ncu_run(func)

            # open the nsight compute ui to visualize the profiling result.
            report.visualize()
    ```

    Parameters
    ----------
    func:
        The function to be profiled.

    args:
        The sequence of arguments to be passed to the function.

    kwargs:
        The dictionary of keyword arguments to be passed to the function.

    Returns
    -------
    NsightComputeReport:
        The report object that can be used to visualize the profiling result.
    """

    # get the python script path and function name
    script_path: str = inspect.getfile(func)
    func_name: str = func.__name__

    # report path
    report_path_template: str = os.path.join(os.path.dirname(script_path), 'ncu-reports/report{}.ncu-rep')
    idx = 0
    while os.path.exists(report_path_template.format(idx)):
        idx += 1
    report_path = report_path_template.format(idx)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    # dump args
    args_path: str = tempfile.mktemp() + '.pkl'
    with open(args_path, 'wb') as f:
        pickle.dump((args, kwargs), f)

    status = subprocess.run(
        _ncu_template.format(
            ncu_path=_ncu_path,
            report_path=report_path,
            python_executable=sys.executable,
            python_script=__file__,
            args='{} {} {}'.format(script_path, func_name, args_path),
        ),
        shell=True,
    )

    if status.returncode != 0:
        raise RuntimeError('Error when running Nsight Compute.')

    return NsightComputeReport(report_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('script_path', type=str)
    parser.add_argument('func', type=str)
    parser.add_argument('args', type=str)
    args = parser.parse_args()
    _ncu_run_func(args.script_path, args.func, args.args)


if __name__ == '__main__':
    main()
