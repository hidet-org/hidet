from typing import Dict, Any, Sequence, Union
import os
import traceback
import sys
import zipfile
import logging
import pickle
from hashlib import sha256
from filelock import FileLock

logger = logging.Logger(__name__)

jobs_dir = os.path.join(os.getcwd(), 'jobs')
repos_dir = os.path.join(os.getcwd(), 'repos')
commits_dir = os.path.join(os.getcwd(), 'commits')
results_dir = os.path.join(os.getcwd(), 'results')
