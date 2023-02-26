import argparse
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, default='.', help='Root directory of the project')

py_license_header = """
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
"""[1:]

cpp_license_header = """
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
"""[1:]

def update_license(root: str, suffix, license_header: str, dry_run: bool = False):
    for path in Path(root).glob('**/*.{}'.format(suffix)):
        with open(path, 'r') as f:
            content = f.read()
            if license_header in content:
                # print('[Skip] {}'.format(path))
                continue
        print('[Update] {}'.format(path))
        content = license_header + content
        if not dry_run:
            with open(path, 'w') as f:
                f.write(content)

def main():
    args = parser.parse_args()
    root = args.root
    update_license(os.path.join(root, 'python'), 'py', py_license_header)
    update_license(os.path.join(root, 'tests'), 'py', py_license_header)
    update_license(os.path.join(root, 'src'), 'cpp', cpp_license_header)
    update_license(os.path.join(root, 'include'), 'h', cpp_license_header)

if __name__ == '__main__':
    main()
