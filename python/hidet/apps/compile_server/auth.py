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
