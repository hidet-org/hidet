import time
from flask import request
from flask_jwt_extended import create_access_token
from flask_restful import Resource
from filelock import FileLock


with FileLock('last_compile_timestamp.txt.lock'):
    with open('last_compile_timestamp.txt', 'r') as f:
        f.write(str(time.time()))


class StatusResource(Resource):
    def get(self):
        query: str = request.json()['query']
        if query == 'last_compile_timestamp':
            with FileLock('last_compile_timestamp.txt.lock'):
                with open('last_compile_timestamp.txt', 'r') as f:
                    return {'timestamp': f.read()}
        else:
            return {'message': 'Invalid query'}, 400
