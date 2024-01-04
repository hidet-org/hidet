import os
import json
import subprocess
import pathlib
import argparse
from tabulate import tabulate

external_models = ['llama-7b', 'gpt2']

def run_command(cmd):
    cmd = " ".join(cmd)
    print("Running command: " + cmd)
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
    stdout, stderr = process.communicate()
    ret = process.returncode
    if ret:
        print('STDERR:')
        for line in stderr:
            print(line, end='')
        raise RuntimeError(f'Command {cmd} failed with return code {ret}.')
    return stdout

def get_bench_cmd(run_type, run_id, run_name, runfile, run_param_name, dtype):
    if run_name in external_models:
        runfile = './models/bench/' + runfile
    else:
        runfile = str(pathlib.Path(__file__).parent.resolve()) + '/bench/' + runfile
    cmd = ['python', runfile, run_name, '--params', run_param_name, '--dtype', dtype]
    return cmd

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Run Benchmarks')
    parser.add_argument(
        '--print',
        action='store_true',
        default=False,
        help='Print results'
    )
    parser.add_argument(
        '--configs',
        type=str,
        default='run_configs.json',
        help='Specify configurations file to use for benchmarking'
    )
    args = parser.parse_args()
    configs_file = args.configs
    fh = open(configs_file)
    run_configs = json.load(fh)
    fh.close()
    hw_config = os.environ.get('HW_CONFIG')
    for run_config in run_configs:
        # Append hardware_config column
        run_config['hardware_config'] = hw_config
        # Extract configurations
        run_type = run_config['type']
        run_id = run_config['id']
        run_name = run_config['name']
        runfile = run_config['runfile']
        run_param_id = run_config['param_id']
        run_param_name = run_config['param_name']
        run_dtype_id = run_config['dtype_id']
        run_dtype_name = run_config['dtype_name']
        cmd = get_bench_cmd(run_type, run_id, run_name, runfile, run_param_name, run_dtype_name)
        outputs = run_command(cmd)
        if outputs:
            # The second last line of All benchmark scripts' stdout is the latency. (Last line is empty)
            latency = float(outputs.split('\n')[-2])
            run_config['latency'] = latency
        else:
            run_config['latency'] = 999.99
    with open(configs_file, 'w') as fh:
        json.dump(run_configs, fh)

    if args.print:
       print(tabulate(run_configs, headers="keys")) 