from flask import request
from flask_jwt_extended import create_access_token
from flask_restful import Resource

from .user import users


class AuthResource(Resource):
    def post(self):
        username = request.json.get('username')
        password = request.json.get('password')

        # Authenticate the user
        if username in users and users[username] == password:
            # Generate an access token
            access_token = create_access_token(identity=username)
            return {'access_token': access_token}
        else:
            return {'message': 'Invalid credentials'}, 401
