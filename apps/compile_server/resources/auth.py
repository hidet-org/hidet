from flask import request
from flask_jwt_extended import create_access_token
from flask_restful import Resource

from .user import users


class AuthResource(Resource):
    def post(self):
        username = request.json.get('username')
        password = request.json.get('password')

        if not isinstance(username, str) or not isinstance(password, str):
            return {'message': 'Invalid credentials'}, 401

        # Authenticate the user
        if username in users:
            if users[username] == password:
                # Generate an access token
                access_token = create_access_token(identity=username, expires_delta=False)
                return {'access_token': access_token}
            else:
                return {'message': 'Invalid credentials'}, 401
        else:
            return {'message': 'Invalid credentials'}, 401
