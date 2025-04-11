"""
Enable python faulthandler to print the python traceback when a segfault occurs.

See: https://docs.python.org/3/library/faulthandler.html
"""

import faulthandler

faulthandler.enable()
