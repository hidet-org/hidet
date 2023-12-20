import os
import time
import subprocess
import json
from db_utils import get_db_conn

def run_command(cmd):
    print("Running command: " + " ".join(cmd))
    output = subprocess.run(cmd, capture_output=True, text=True)
    print(output.stdout)
    print(output.stderr)
    return output


if __name__ == '__main__':
    conn = get_db_conn()
    cursor = conn.cursor()

    # e.g., ' 1, 2, ,3,,' -> ['1', '2', '3']
    hw_config_ids = os.environ.get('HW_CONFIG').replace(' ', '')
    repo_org = os.environ.get('REPO_NAME').split('/')[0]
    if hw_config_ids == 'all':
        query = (
            'SELECT id FROM hardware_config'
        )
        cursor.execute(query)
        rows = cursor.fetchall()
        hw_config_ids = [row[0] for row in rows]
    else:
        hw_config_ids = [s for s in hw_config_ids.split(',') if s]

    instances = []
    # Fetch list of (cloud_provider_id, instance_id) tuples from DB
    for hw_config_id in hw_config_ids:
        query = (
            'SELECT cloud_provider_id, instance_id, hardware_config.name as hw_config FROM cloud_instance '
            'JOIN hardware_config ON cloud_instance.hardware_config_id = hardware_config.id '
            f'WHERE hardware_config_id = {hw_config_id} AND cloud_instance.org = \'{repo_org}\' LIMIT 1'
        )
        cursor.execute(query)
        rows = cursor.fetchall()
        if len(rows) == 0:
            raise ValueError(f'Instance with hardware config id {hw_config_id} does not exist.')
        instances.append(rows[0])

    # Store a json containing all the required model/OPs (and inputs) for this regression run
    # This json will be uploaded as an artifact, and will be filled in by subsequent jobs
    # For now, we run all model/input combinations by default
    run_configs = []
    query = (
        'SELECT model.id as model_id, model.name as model_name, input_parameter.id as param_id, '
        'input_parameter.parameter as param_name, dtype.id as dtype_id, dtype.name as dtype_name '
        'FROM model JOIN model_input_parameter ON '
        'model.id = model_input_parameter.model_id JOIN input_parameter ON '
        'model_input_parameter.input_parameter_id = input_parameter.id JOIN dtype'
    )
    cursor.execute(query)
    rows = cursor.fetchall()
    for row in rows:
        model_id, model_name, param_id, param_name, dtype_id, dtype_name = row
        run_configs.append({'type': 'model', 'id': int(model_id), 'name': model_name, 
                            'param_id': int(param_id), 'param_name': param_name,
                            'dtype_id': int(dtype_id), 'dtype_name': dtype_name,
                            })
    query = (
        'SELECT operator.id as operator_id, operator.name as operator_name, input_parameter.id as param_id, '
        'input_parameter.parameter as param_name, dtype.id as dtype_id, dtype.name as dtype_name '
        'FROM operator JOIN operator_input_parameter ON '
        'operator.id = operator_input_parameter.operator_id JOIN input_parameter ON '
        'operator_input_parameter.input_parameter_id = input_parameter.id JOIN dtype'
    )
    cursor.execute(query)
    rows = cursor.fetchall()
    for row in rows:
        op_id, op_name, param_id, param_name, dtype_id, dtype_name = row
        run_configs.append({'type': 'operator', 'id': int(op_id), 'name': op_name, 
                            'param_id': int(param_id), 'param_name': param_name,
                            'dtype_id': int(dtype_id), 'dtype_name': dtype_name,
                            })
    with open('run_configs.json', 'w') as fh:
        json.dump(run_configs, fh)

    # Close DB connection
    cursor.close()
    conn.close()

    # Start all instances
    for instance in instances:
        cloud_provider_id, instance_id, _ = instance
        if cloud_provider_id == 1: # AWS
            cmd = ['aws', 'ec2', 'start-instances', '--instance-ids', instance_id]
        else:
            raise ValueError(f'Unknown cloud provider id: {cloud_provider_id}')
        output = run_command(cmd)
        if output.returncode != 0:
            raise RuntimeError(f'Failed to start instance {instance_id} on cloud provider {cloud_provider_id}.')

    # Wait until all instances are running
    for instance in instances:
        cloud_provider_id, instance_id, _ = instance
        started = False
        while not started:
            time.sleep(5)
            if cloud_provider_id == 1: # AWS
                cmd = ['aws', 'ec2', 'describe-instance-status', '--instance-ids', instance_id]
                output = run_command(cmd)
                if output.returncode != 0:
                    raise RuntimeError(f'Failed to check status for {instance_id} on cloud provider {cloud_provider_id}.')
                if output.stdout.count('ok') >= 2:
                    started = True
            else:
                raise ValueError(f'Unknown cloud provider id: {cloud_provider_id}')

    # Set outputs for subsequent jobs to use

    # String representing launched instances
    # e.g., "1,aws-instance0;1,aws-instance1;2,gcp-instance0" representing two AWS instances and one GCP instance
    instances_str = ''
    for instance in instances:
        cloud_provider_id, instance_id, _ = instance
        instances_str += f'{cloud_provider_id},{instance_id};'
    with open(os.environ['GITHUB_OUTPUT'], 'a') as fh:
        print(f'started_instances={instances_str}', file=fh)

    # String representing JSON Array of hardware configs of the runners
    hw_configs = []
    for instance in instances:
        _, _, hw_config = instance
        hw_configs.append(hw_config)
    hw_config_json_str = json.dumps(hw_configs)
    with open(os.environ['GITHUB_OUTPUT'], 'a') as fh:
        print(f'hw_configs={hw_config_json_str}', file=fh)