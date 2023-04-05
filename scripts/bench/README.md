# Benchmark Server

This script is used to benchmark the performance of hidet, repeatedly running and send github comment to an issue.

## Usage

```bash
# Install github cli
sudo apt install -y gh
# clone the repo
git clone git@github.com:hidet-org/hidet
# cd into the repo
cd hidet
# run the daemon script, you can specify the issue to send the report to
python scripts/bench/run.py [--issue <issue>]
```
