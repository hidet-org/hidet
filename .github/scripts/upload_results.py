import os
import json
from db_utils import get_db_conn

if __name__ == '__main__':
    # Get environment variables
    commit_sha = os.environ.get('COMMIT_SHA')
    commit_time = os.environ.get('COMMIT_TIME')
    commit_author = os.environ.get('COMMIT_AUTHOR')
    repo_name = os.environ.get('REPO_NAME').strip()
    repo_branch = os.environ.get('REPO_BRANCH').strip()
    hw_configs = os.environ.get('HW_CONFIGS')
    if 'pull/' in repo_branch:
        commit_url = f'https://github.com/{repo_name}/{repo_branch}/commits/{commit_sha}'
    else:
        commit_url = f'https://github.com/{repo_name}/commit/{commit_sha}'

    # Insert commit into DB
    conn = get_db_conn()
    cursor = conn.cursor()

    query = (
        'INSERT INTO commit (hash, url, author, time, status, branch) VALUES (%s, %s, %s, %s, %s, %s)'
    )
    val = (commit_sha[:7], commit_url, commit_author, commit_time, 'pass', repo_name + '/' + repo_branch)
    cursor.execute(query, val)
    conn.commit()

    query = ('SELECT LAST_INSERT_ID()')
    cursor.execute(query)
    commit_id = cursor.fetchall()[0][0]

    # Create a mapping of HW config name to HW config ID
    query = (
        'SELECT id, name FROM hardware_config'
    )
    cursor.execute(query)
    hw_config_table = cursor.fetchall()
    hw_config_map = {}
    for hw_config in hw_config_table:
        hw_config_map[hw_config[1]] = int(hw_config[0])

    # Insert results into table
    hw_configs = json.loads(hw_configs)
    for hw_config in  hw_configs:
        artifact_path = f'./run_configs_{hw_config}/run_configs.json'
        fh = open(artifact_path)
        run_configs = json.load(fh)
        fh.close()
        for run_config in run_configs:
            run_type = run_config['type']
            run_id = run_config['id']
            run_param_id = run_config['param_id']
            run_dtype_id = run_config['dtype_id']
            run_hw_config = run_config['hardware_config'] # Should be same as `hw_config`
            run_latency = run_config['latency']
            run_hw_config_id = hw_config_map[run_hw_config]
            query = (
                f'INSERT INTO {run_type}_result (commit_id, {run_type}_id, input_parameter_id, hardware_config_id, '
                f'dtype_id, latency) VALUES (%s, %s, %s, %s, %s, %s)'
            )
            val = (commit_id, run_id, run_param_id, run_hw_config_id, run_dtype_id, run_latency)
            cursor.execute(query, val)
            conn.commit()