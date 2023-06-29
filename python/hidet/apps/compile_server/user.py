import requests
from .core import access_token, api_url


def add_user(username, password):
    response = requests.post(
        api_url('user'), json={
            'username': username,
            'password': password
        }, headers={
            'Authorization': f'Bearer {access_token()}'
        }
    )
    if response.status_code != 201:
        print('Error: ', response.json()['message'])
        return
    print(response.json()['message'])


def delete_user(username):
    response = requests.delete(
        api_url('user'), json={
            'username': username,
        }, headers={
            'Authorization': f'Bearer {access_token()}'
        }
    )
    if response.status_code != 200:
        print('Error: ', response.json()['message'])
        return
    print(response.json()['message'])
