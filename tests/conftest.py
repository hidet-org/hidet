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
import os
import pytest
import shutil
import hidet


def pytest_addoption(parser):
    parser.addoption("--clear-cache", action="store_true", help="Clear operator cache before running tests")


def pytest_sessionstart(session):
    """
    Called after the Session object has been created and before performing collection and entering the run test loop.
    """
    # set the cache directory to a subdirectory of the current directory
    hidet.option.cache_dir(os.path.join(hidet.option.get_cache_dir(), 'test_cache'))
    print('Cache directory: {}'.format(hidet.option.get_cache_dir()))

    if session.config.getoption("--clear-cache"):
        print('Clearing cache directory: {}'.format(hidet.option.get_cache_dir()))

        # clean the operator cache directory
        print('Clearing operator cache in test cache...')
        hidet.utils.hidet_clear_op_cache()
