import subprocess
import sys
import os


def test_option_env():
    env_name = 'HIDET_NUM_WORKERS'
    expected = 4

    # create a new python process with the environment variable set, using spawn context
    env = os.environ.copy()
    env[env_name] = str(expected)

    # launch the process
    command = '{python} {script} --expected {expected}'.format(
        python=sys.executable, script=os.path.join(os.path.dirname(__file__), 'main.py'), expected=expected
    )
    subprocess.run(command, shell=True, env=env, check=True)
