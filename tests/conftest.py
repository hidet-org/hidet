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
