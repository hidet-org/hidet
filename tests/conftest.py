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
    parser.addoption("--runslow", action="store_true", help="Run slow tests")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


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
        print('Clearing hidet cache in test cache...')
        hidet.utils.clear_cache_dir('ops')
        hidet.utils.clear_cache_dir('testing')
        hidet.utils.clear_cache_dir('graphs')


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture(autouse=True)
def clear_memory_cache():
    """
    Clear the memory cache before each test.
    """
    # run before each test
    import torch

    torch.cuda.empty_cache()
    hidet.runtime.storage.current_memory_pool('cuda').clear()
    yield
    # run after each test
    pass
