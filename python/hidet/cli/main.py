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
from hidet.cli.bench import bench_group
from hidet.utils import initialize


@click.group(name='hidet')
def main():
    pass


@initialize()
def register_commands():
    for group in [bench_group]:
        assert isinstance(group, click.Command)
        main.add_command(group)


if __name__ == '__main__':
    main()
