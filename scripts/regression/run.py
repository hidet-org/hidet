import os
import time
import argparse
import subprocess
import schedule

parser = argparse.ArgumentParser(prog='Performance Regression Scheduler',
                                 description='This script will periodically launch'
                                 ' a performance regression every Friday at 10pm.')
parser.add_argument(
    '--now',
    action='store_true',
    help='Launch a regression immediately and return.'
)


def install_dependencies():
    subprocess.run(['pip', 'install', '-r', 'requirements.txt'], check=True)
    subprocess.run(['pip', 'install', '-r', 'requirements-dev.txt'], check=True)
    subprocess.run(['pip', 'install', '-r', 'scripts/regression/requirements.txt'], check=True)


def pull_repo():
    subprocess.run(['git', 'pull'], check=True)


def reinstall_hidet():
    subprocess.run(['pip', 'uninstall', 'hidet', '-y'], check=True)
    subprocess.run(['mkdir', '-p', 'build'], check=True)
    subprocess.run(['rm', '-rf', 'build/*'], check=True)
    subprocess.run(['cmake', '-S', '.', '-B', 'build'], check=True)
    subprocess.run(['cmake', '--build', 'build'], check=True)
    subprocess.run(['pip', 'install', '-e', '.'], check=True)


def run_regression(report_file):
    from model_performance import model_performance_regression
    from op_performance import op_performance_regression
    model_performance_regression(report_file)
    op_performance_regression(report_file)
    return
    space = 2
    current_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').strip()
    command = 'python scripts/bench/benchmark.py --git-commit {commit} --space {space} --report {report_file}'.format(
        commit=current_commit, report_file=report_file, space=space
    )

    if os.path.exists('scripts/bench/prev_commit.txt'):
        with open('scripts/bench/prev_commit.txt', 'r') as f:
            prev_commit = f.readline().strip()
        command += ' --git-prev-commit {}'.format(prev_commit)

    subprocess.run(command.split(), check=True)

    with open('scripts/bench/prev_commit.txt', 'w') as f:
        f.write(current_commit)


def send_report(result_file):
    command = 'ls'
    subprocess.run(command.split(), check=True)


def bench_job():
    report_file = './scripts/regression/report.txt'
    try:
        pull_repo()
        reinstall_hidet()
        run_regression(report_file)
        send_report(report_file)
    except Exception as e:
        print('Error: {}'.format(e))


def main():
    args = parser.parse_args()
    if not os.path.exists('./scripts/regression/run.py'):
        raise RuntimeError('Please run this script from the root directory of the repository.')

    install_dependencies()

    now = args.now
    if now:
        bench_job()
        return

    schedule.every().friday.at("22:00").do(bench_job)

    while True:
        schedule.run_pending()
        time.sleep(60)


if __name__ == '__main__':
    main()

