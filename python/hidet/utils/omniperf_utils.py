# pylint: disable=subprocess-run-check
import os
import pickle
import argparse
import sys
import subprocess


class OmniperfReport:
    def __init__(self, report_path: str, omniperf_path: str):
        self.report_path: str = report_path
        self.omniperf_path: str = omniperf_path

    def visualize(self):
        subprocess.run(f"{self.omniperf_path} analyze -p {self.report_path} --gui", shell=True)


def __run_func(script_path, func_name, args_pickled_path):
    with open(args_pickled_path, 'rb') as f:
        args, kwargs = pickle.load(f)
    print("running func")
    try:
        sys.path.append(os.path.dirname(script_path))
        module = __import__(os.path.basename(script_path)[:-3])
    except Exception as e:
        raise RuntimeError('Can not import the python script: {}'.format(script_path)) from e

    if not hasattr(module, func_name):
        raise RuntimeError('Can not find the function "{}" in {}'.format(func_name, script_path))

    func = getattr(module, func_name)
    print(func)

    try:
        func(*args, **kwargs)
    except Exception as e:
        raise RuntimeError('Error when running the function "{}"'.format(func_name)) from e


def omniperf_run(omniperf_path, func, *args, **kwargs) -> OmniperfReport:
    import inspect
    import tempfile

    # get the python script path and function name
    script_path: str = inspect.getfile(func)
    func_name: str = func.__name__

    # report path
    report_path_template: str = os.path.join(os.path.dirname(script_path), 'omniperf_data/experiment{}')
    idx = 0
    while os.path.exists(report_path_template.format(idx)):
        idx += 1
    report_path = report_path_template.format(idx)
    os.makedirs(report_path)

    # dump args
    with tempfile.NamedTemporaryFile('wb', suffix='pkl') as f:
        args_path: str = f.name
        pickle.dump((args, kwargs), f)

        launch_file: str = report_path + "/launch.sh"
        launch_command = f"{sys.executable} {str(__file__)} {str(script_path)} {str(func_name)} {str(args_path)}\n"
        assert os.path.exists(script_path)
        assert os.path.exists(args_path)
        assert os.path.exists(__file__)

        print("writing launch file: ", launch_command)
        with open(launch_file, 'w') as f:
            f.write(launch_command)

        omniperf_data_dir = os.path.join(report_path, "data")
        os.mkdir(omniperf_data_dir)
        print(f"chmod +x {launch_file}")
        subprocess.run(f"chmod +x {launch_file}", shell=True)
        print(f"prerunning {launch_file}")
        subprocess.run(launch_file, shell=True)
        print(f"{omniperf_path} profile -n experiment{idx} -p {omniperf_data_dir} -- {launch_file}")
        subprocess.run(
            f"{omniperf_path} profile -n experiment{idx} -p {omniperf_data_dir} -- {launch_file}", shell=True
        )

        return OmniperfReport(omniperf_data_dir, omniperf_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('script_path', type=str)
    parser.add_argument('func', type=str)
    parser.add_argument('args', type=str)
    args = parser.parse_args()
    __run_func(args.script_path, args.func, args.args)


if __name__ == '__main__':
    main()
