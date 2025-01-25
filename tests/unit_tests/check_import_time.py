import time

# first import torch so that we only profile the time used to import hidet itself (and its other dependencies)
# torch takes more than 0.5 seconds when I add this test, and it is not ignorable.
import torch

t1 = time.time_ns()
import hidet

t2 = time.time_ns()

import_time = (t2 - t1) / 1e9
print('Import hidet takes: {:.3f} seconds'.format(import_time))
assert import_time < 2.0  # make sure hidet could be imported within (1 seconds + torch's import time)
