import os
import json
import subprocess
import pathlib
import numpy as np
import tqdm
from db_utils import get_db_conn

external_models = ['llama-7b', 'gpt2']

def run_command(cmd):
    cmd = " ".join(cmd)
    print("Running command: " + cmd)
    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
    outputs = []
    for line in popen.stdout:
        print(line, end='')
        outputs.append(line)
    popen.stdout.close()
    ret = popen.wait()
    if ret:
        print('STDERR:')
        for line in popen.stderr:
            print(line, end='')
        print(f'Command {cmd} failed with return code {ret}.')
        return None
    return outputs

def get_bench_cmd(run_type, run_id, run_name, run_param_name, dtype):
    # Get the name of the benchmark script from DB
    conn = get_db_conn()
    cursor = conn.cursor()
    query = f'SELECT runfile FROM {run_type} WHERE id = {run_id}'
    cursor.execute(query)
    runfile = cursor.fetchall()[0][0]
    cursor.close()
    conn.close()
    if run_name in external_models:
        runfile = './models/bench/' + runfile
    else:
        runfile = str(pathlib.Path(__file__).parent.resolve()) + '/bench/' + runfile
    cmd = ['python', runfile, run_name, '--params', run_param_name, '--dtype', dtype]
    return cmd

if __name__ == '__main__':
    fh = open('run_configs.json')
    run_configs = json.load(fh)
    fh.close()
    hw_config = os.environ.get('HW_CONFIG')
    print('hw:', hw_config)
    for run_config in run_configs:
        # Append hardware_config column
        run_config['hardware_config'] = hw_config
        # Extract configurations
        run_type = run_config['type']
        run_id = run_config['id']
        run_name = run_config['name']
        run_param_id = run_config['param_id']
        run_param_name = run_config['param_name']
        run_dtype_id = run_config['dtype_id']
        run_dtype_name = run_config['dtype_name']
        cmd = get_bench_cmd(run_type, run_id, run_name, run_param_name, run_dtype_name)
        outputs = run_command(cmd)
        if outputs:
            latency = float(outputs[-1].split('\n')[0]) # Get last line
            run_config['latency'] = latency
        else:
            run_config['latency'] = 999.99
    with open('run_configs.json', 'w') as fh:
        json.dump(run_configs, fh)