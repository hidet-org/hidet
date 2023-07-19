import os
import time
import argparse
import subprocess
import schedule
from email_sender import EmailSender

parser = argparse.ArgumentParser(prog='Performance Regression Scheduler',
                                 description='This script will periodically launch'
                                 ' a performance regression every Friday at 10pm.')
parser.add_argument(
    '--now',
    action='store_true',
    help='Launch a regression immediately and return.'
)
parser.add_argument(
    '--email',
    action='store_true',
    help='Send results to email. Requires Gmail login via app password.'
)
parser.add_argument(
    '--keep',
    action='store_true',
    help='Keep operator cache.'
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


def run_regression(report_file, keep_cache=False):
    if not keep_cache:
        command = f'hidet cache clear'
        subprocess.run(command.split(), check=True)

    model_report_file = './scripts/regression/report_model_performance.txt'
    op_report_file = './scripts/regression/report_op_performance.txt'

    command = f'python scripts/regression/model_performance.py --report {model_report_file}'
    subprocess.run(command.split(), check=True)
    command = f'python scripts/regression/op_performance.py --report {op_report_file}'
    subprocess.run(command.split(), check=True)

    # Merge report files into one
    model_report = op_report = ""
    with open(model_report_file, 'r') as f:
        model_report = f.read()
    with open(op_report_file, 'r') as f:
        op_report = f.read()
    report = model_report + '\n' + op_report
    with open(report_file, 'w') as f:
        f.write(report)
    

def bench_job(sender, keep_cache=False):
    report_file = './scripts/regression/report.txt'
    try:
        pull_repo()
        reinstall_hidet()
        run_regression(report_file, keep_cache=keep_cache)
        with open(report_file, 'r') as f:
            report = f.read()
            print(report)
            if sender is not None:
                sender.send_email(report)
    except Exception as e:
        print('Error: {}'.format(e))


def main():
    args = parser.parse_args()
    if not os.path.exists('./scripts/regression/run.py'):
        raise RuntimeError('Please run this script from the root directory of the repository.')

    install_dependencies()
    print("Finished installing dependencies.")

    sender = EmailSender() if args.email else None

    print("Waiting for next scheduled run.")

    now = args.now
    keep_cache = args.keep
    if now:
        bench_job(sender, keep_cache=keep_cache)
        return

    schedule.every().friday.at("22:00").do(bench_job, sender, keep_cache)

    while True:
        schedule.run_pending()
        time.sleep(60)


if __name__ == '__main__':
    main()

