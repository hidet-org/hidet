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
    command = f'python scripts/regression/model_performance.py --report {report_file}'
    subprocess.run(command.split(), check=True)
    command = f'python scripts/regression/op_performance.py --report {report_file}'
    subprocess.run(command.split(), check=True)

def bench_job(sender):
    report_file = './scripts/regression/report.txt'
    if os.path.exists(report_file):
        os.remove(report_file)
    try:
        pull_repo()
        reinstall_hidet()
        run_regression(report_file)
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
    if now:
        bench_job(sender)
        return

    schedule.every().friday.at("22:00").do(bench_job, sender)

    while True:
        schedule.run_pending()
        time.sleep(60)


if __name__ == '__main__':
    main()

