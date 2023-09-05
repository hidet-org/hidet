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
import requests
from .core import api_url


def get_access_token(username, password):
    try:
        response = requests.post(api_url('auth'), json={'username': username, 'password': password})
    except requests.exceptions.ConnectionError:
        raise RuntimeError('Can not connect to compiler server {}'.format(api_url(''))) from None

    if response.status_code != 200:
        raise RuntimeError('Failed to get access token: {}'.format(response.json()['message']))
    return response.json()['access_token']
