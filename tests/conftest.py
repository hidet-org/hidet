import os
import shutil
import hidet


def pytest_sessionstart(session):
    """
    Called after the Session object has been created and before performing collection and entering the run test loop.
    """
    # set the cache directory to a subdirectory of the current directory
    hidet.option.cache_dir(os.path.join(hidet.option.get_cache_dir(), 'test_cache'))

    # clean the operator cache directory
    print('Clearing operator cache in test cache...')
    hidet.utils.hidet_clear_op_cache()
