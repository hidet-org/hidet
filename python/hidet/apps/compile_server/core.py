from typing import Optional
import hidet

_api_url: Optional[str] = None
_access_token: Optional[str] = None


def init_api():
    global _api_url, _access_token
    from .auth import get_access_token

    _api_url = 'http://{}:{}'.format(
        hidet.option.get_option('compile_server.addr'),
        hidet.option.get_option('compile_server.port')
    )
    username = hidet.option.get_option('compile_server.username'),
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

