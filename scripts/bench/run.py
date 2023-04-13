import os
import time
import argparse
import subprocess
import schedule

parser = argparse.ArgumentParser('Benchmark performance.')
parser.add_argument('--issue', type=int, default=154, help='Issue id to send the benchmark result to.')
parser.add_argument('--space', type=int, default=2, help='Search space.')
parser.add_argument(
    '--schedule-time',
    type=str,
    default='03:00',
    help='Schedule time to run the benchmark, default "3:00".'
)


def install_dependencies():
    subprocess.run(['pip', 'install', '-r', 'requirements.txt'], check=True)
    subprocess.run(['pip', 'install', '-r', 'requirements-dev.txt'], check=True)
    subprocess.run(['pip', 'install', '-r', 'scripts/bench/requirements.txt'], check=True)


def pull_repo():
    subprocess.run(['git', 'pull'], check=True)


def reinstall_hidet():
    subprocess.run(['pip', 'uninstall', 'hidet', '-y'], check=True)
    subprocess.run(['mkdir', '-p', 'build'], check=True)
    subprocess.run(['rm', '-rf', 'build/*'], check=True)
    subprocess.run(['cmake', '-S', '.', '-B', 'build'], check=True)
    subprocess.run(['cmake', '--build', 'build'], check=True)
    subprocess.run(['pip', 'install', '-e', '.'], check=True)


def run_bench_script(space, report_file):
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


def send_report(issue, result_file):
    command = 'gh issue comment {issue} -F {result_file} -R hidet-org/hidet'.format(
        issue=issue, result_file=result_file
    )
    subprocess.run(command.split(), check=True)


def bench_job(args):
    report_file = './scripts/bench/report.txt'
    try:
        pull_repo()
        reinstall_hidet()
        run_bench_script(args.space, report_file)
        send_report(args.issue, report_file)
    except Exception as e:
        print('Error: {}'.format(e))


def main():
    args = parser.parse_args()
    if not os.path.exists('./scripts/bench/benchmark.py'):
        raise RuntimeError('Please run this script from the root directory of the repository.')
    if not (0 <= int(args.schedule_time.split(':')[0]) <= 23 and 0 <= int(args.schedule_time.split(':')[1]) <= 59):
        raise RuntimeError('Schedule time should be in the format of "HH:MM".')

    install_dependencies()

    print(args.schedule_time)
    schedule.every().day.at(args.schedule_time).do(bench_job, args=args)

    while True:
        schedule.run_pending()
        time.sleep(1)


if __name__ == '__main__':
    main()

