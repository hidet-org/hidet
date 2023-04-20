# Benchmark Server

This script is used to benchmark the performance of hidet, repeatedly running and send github comment to an issue.

## Usage

```bash
# Install github cli
sudo apt install -y gh
# Install dependency
pip install -r scripts/bench/requirements.txt
# clone the repo
git clone git@github.com:hidet-org/hidet
# cd into the repo
cd hidet
# run the daemon script, you can specify the issue to send the report to
# <issue> is the issue number, e.g. 135
# <schedule-time> is the time to run the benchmark everyday in format HH:MM, e.g. 03:00
python scripts/bench/run.py [--issue <issue>] [--schedule-time <schedule-time>]
```
