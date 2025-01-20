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
import gc
from contextlib import contextmanager


@contextmanager
def gc_disabled():
    """
    A context manager to disable garbage collection
    and ensure it is re-enabled afterward.
    """
    was_enabled = gc.isenabled()
    gc.disable()  # Disable garbage collection
    try:
        yield
    finally:
        if was_enabled:
            gc.enable()  # Re-enable garbage collection if it was originally enabled
