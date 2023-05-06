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
import click
from .status import hidet_cache_status
from .clear import hidet_cache_clear


@click.group(name='cache', help='Manage hidet cache.')
def hidet_cache_group():
    pass


for command in [hidet_cache_status, hidet_cache_clear]:
    assert isinstance(command, click.Command)
    hidet_cache_group.add_command(command)
