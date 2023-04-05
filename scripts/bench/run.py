import os
import time
import argparse
import subprocess

parser = argparse.ArgumentParser('Benchmark performance.')
parser.add_argument('--issue-id', type=int, default=154, help='Issue id to send the benchmark result to.')

def install_dependencies():
    subprocess.run(['pip', 'install', '-r', 'requirements.txt'], check=True)
    subprocess.run(['pip', 'install', '-r', 'requirements-dev.txt'], check=True)

def pull_repo():
    subprocess.run(['git', 'pull'], check=True)


def reinstall_hidet():
    subprocess.run(['pip', 'uninstall', 'hidet', '-y'], check=True)
    subprocess.run(['mkdir', '-p', 'build'], check=True)
    subprocess.run(['rm', '-rf', 'build/*'], check=True)
    subprocess.run(['cmake', '-S', '.', '-B', 'build'], check=True)
    subprocess.run(['pip', 'install', '-e', '.'], check=True)


def run_bench_script(report_file):
    current_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').strip()
    command = 'python scripts/bench/benchmark.py --git-commit {commit} --report {report_file}'.format(
        commit=current_commit, report_file=report_file
    )

    if os.path.exists('scripts/bench/prev_commit.txt'):
        with open('scripts/bench/prev_commit.txt', 'r') as f:
            prev_commit = f.readline().strip()
        command += ' --git-prev-commit {}'.format(prev_commit)

    subprocess.run(command.split(), check=True)

    with open('scripts/bench/prev_commit.txt', 'w') as f:
        f.write(current_commit)

def send_report(issue_id, result_file):
    command = 'gh issue comment {issue_id} -F {result_file} -R hidet-org/hidet'.format(
        issue_id=issue_id, result_file=result_file
    )
    subprocess.run(command.split(), check=True)


def main():
    args = parser.parse_args()
    if not os.path.exists('./scripts/bench/benchmark.py'):
        raise RuntimeError('Please run this script from the root directory of the repository.')

    install_dependencies()

    while True:
        t1 = time.time()
        try:
            report_file = './scripts/bench/report.txt'
            pull_repo()
            reinstall_hidet()
            run_bench_script(report_file)
            send_report(args.issue_id, report_file)
        except Exception as e:
            print('Error: {}'.format(e))
            time.sleep(60 * 60) # wait for 1 hour
        else:
            # run the benchmark once every day
            t2 = time.time()
            print('Elapsed time: {} seconds'.format(t2 - t1))
            if t2 - t1 < 60 * 60 * 24:
                time.sleep(60 * 60 * 24 - (t2 - t1))


if __name__ == '__main__':
    main()

