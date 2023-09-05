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
from typing import Optional
import hidet

_api_url: Optional[str] = None
_access_token: Optional[str] = None


def init_api():
    global _api_url, _access_token
    from .auth import get_access_token

    _api_url = 'http://{}:{}'.format(
        hidet.option.get_option('compile_server.addr'), hidet.option.get_option('compile_server.port')
    )
    username = hidet.option.get_option('compile_server.username')
    password = hidet.option.get_option('compile_server.password')
    _access_token = get_access_token(username, password)


def api_url(resource: str):
    if _api_url is None:
        init_api()
    return f'{_api_url}/{resource}'


def access_token():
    if _access_token is None:
        init_api()
    return _access_token
