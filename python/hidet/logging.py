# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Optional
import logging

logger = logging.getLogger('hidet')
stderr_handler = logging.StreamHandler()  # defaults to sys.stderr


def init_hidet_root_logger():
    logger.setLevel(logging.DEBUG)
    stderr_handler.setLevel(logging.WARNING)

    console_formatter = logging.Formatter(fmt='%(name)s: [%(levelname)s] %(message)s')
    stderr_handler.setFormatter(console_formatter)
    logger.addHandler(stderr_handler)


init_hidet_root_logger()


def to_file(filename: str, level: Optional[int] = logging.DEBUG):
    """
    Add a file handler to the hidet root logger

    Parameters
    ----------
    filename: str
        The file to write to.

    level: int, optional
        The logging level.
    """
    file_formatter = logging.Formatter(fmt='%(asctime)s %(name)s: [%(levelname)s] %(message)s')
    handler = logging.FileHandler(filename)
    handler.setFormatter(file_formatter)
    handler.setLevel(level)
    logger.addHandler(handler)


def setConsoleLevel(level: int):
    """
    Set the logging level of the console handler (to stderr) in the hidet root logger

    Parameters
    ----------
    level: int
        The logging level. Can be one of the following:
        - logging.DEBUG - detailed information, typically of interest only when diagnosing problems.
        - logging.INFO - confirmation that things are working as expected.
        - logging.WARNING - an indication that something unexpected happened.
    """
    stderr_handler.setLevel(level)
