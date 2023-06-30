import os
import subprocess
import time
import requests
from subprocess import Popen, PIPE
from filelock import FileLock

def run(command: str, cwd: str = None):
    subprocess.run(command.split(), cwd=cwd, check=True)

def update_hidet():
    if not os.path.exists('hidet'):
        run('git clone https://github.com/yaoyaoding/hidet.git')
        run('git checkout ccapp', cwd='./hidet')
    run('git pull', cwd='./hidet')
    run('mkdir -p build', cwd='./hidet')
    run('cmake ..', cwd='./hidet/build')
    run('make -j4', cwd='./hidet/build')
    run('pip install --force-reinstall -e .', cwd='./hidet')
    run('pip install -r requirements.txt', cwd='./hidet/apps/compile_server')


def keep_idle_for_30minutes():
    path = './hidet/apps/compile_server/last_compile_timestamp.txt'
    with FileLock(path + '.lock'):
        if os.path.exists(path):
            with open(path, 'r') as f:
                last_compile_timestamp = float(f.read())
                if time.time() - last_compile_timestamp > 30 * 60:
                    return True
    return False


def main():
    update_hidet()
    cmd ='gunicorn -w {} -b 0.0.0.0:3281 app:app'.format(os.cpu_count())
    proc = Popen(cmd.split(), stdout=PIPE, stdin=PIPE, cwd='./hidet/apps/compile_server')

    while True:
        ret = proc.poll()
        if ret is not None:
            print('Server is down, restarting...')
            proc = Popen(cmd.split(), stdout=PIPE, stdin=PIPE, cwd='./hidet/apps/compile_server')
        if keep_idle_for_30minutes():
            print('Idle for 30 minutes, try to update hidet to latest version...')
            proc.terminate()
            update_hidet()
            proc = Popen(cmd.split(), stdout=PIPE, stdin=PIPE, cwd='./hidet/apps/compile_server')
        time.sleep(5)   # check every 5 seconds

if __name__ == '__main__':
    main()
