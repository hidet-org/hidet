import os
import subprocess
import sys


def run(command: str, cwd: str = None, env=None):
    subprocess.run(command.split(), cwd=cwd, check=True, env=env)


def main():
    os.makedirs(name='compile_server/', exist_ok=True)
    os.makedirs(name='compile_server/jobs/', exist_ok=True)
    os.makedirs(name='compile_server/repos/', exist_ok=True)
    os.makedirs(name='compile_server/commits/', exist_ok=True)
    os.makedirs(name='compile_server/results/', exist_ok=True)

    # download the compile server code from official repo
    if not os.path.exists('compile_server/hidet'):
        run('git clone https://github.com/hidet-org/hidet.git compile_server/hidet')
    else:
        run('git pull', cwd='compile_server/hidet')

    # install the dependencies
    run('pip install -r requirements.txt', cwd='compile_server/hidet/apps/compile_server')

    # run the compile server
    debug = False
    if debug:
        run(
            '{} ./hidet/apps/compile_server/app.py'.format(sys.executable),
            cwd=os.path.abspath('./compile_server'),
        )
    else:
        run(
            'gunicorn -w {} -b 0.0.0.0:3281 --pythonpath {} app:app --capture-output --timeout {}'.format(
                os.cpu_count(),
                os.path.abspath('./compile_server/hidet/apps/compile_server'),
                60 * 60  # 1 hour
            ),
            cwd=os.path.abspath('./compile_server')
        )


if __name__ == '__main__':
    main()
