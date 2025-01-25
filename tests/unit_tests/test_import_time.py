import os
import sys
import subprocess


def test_import_time():
    """
    Make sure hidet can be imported within a given time limit.
    """
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'check_import_time.py')
    subprocess.check_call([sys.executable, script_path])
