import pytest
import os
import sys
import subprocess


def test_lazy_initialization():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    python_path = sys.executable
    cmd = [python_path, os.path.join(cur_dir, 'lazy_init_sample.py')]
    subprocess.run(cmd, check=True)


if __name__ == '__main__':
    pytest.main([__file__])
