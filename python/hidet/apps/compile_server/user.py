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
from .core import access_token, api_url


def add_user(username, password):
    response = requests.post(
        api_url('user'),
        json={'username': username, 'password': password},
        headers={'Authorization': f'Bearer {access_token()}'},
    )
    if response.status_code != 201:
        print('Error: ', response.json()['message'])
        return
    print(response.json()['message'])


def delete_user(username):
    response = requests.delete(
        api_url('user'), json={'username': username}, headers={'Authorization': f'Bearer {access_token()}'}
    )
    if response.status_code != 200:
        print('Error: ', response.json()['message'])
        return
    print(response.json()['message'])
