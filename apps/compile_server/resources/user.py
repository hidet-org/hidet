import os
from flask import request
from flask_jwt_extended import jwt_required, get_jwt_identity
from flask_restful import Resource

admin_passwd = os.environ.get('HIDET_CF_ADMIN_PASSWD', 'admin_password')

users = {
    'admin': admin_passwd,
}


class UserResource(Resource):
    @jwt_required()
    def get(self):
        current_user = get_jwt_identity()

        if current_user != 'admin':
            return {'message': 'You are not authorized to access this resource'}, 403
        return users

    @jwt_required()
    def post(self):
        current_user = get_jwt_identity()

        if current_user != 'admin':
            return {'message': 'You are not authorized to access this resource'}, 403

        username = request.json.get('username')
        password = request.json.get('password')

        if username in users:
            return {'message': 'Username already exists'}, 409
        users[username] = password
        return {'message': 'User created successfully'}, 201

    @jwt_required()
    def delete(self):
        current_user = get_jwt_identity()

        if current_user != 'admin':
            return {'message': 'You are not authorized to access this resource'}, 403

        username = request.json.get('username')

        if username not in users:
            return {'message': 'User not found'}, 404
        del users[username]
        return {'message': 'User deleted successfully'}, 200

    @jwt_required()
    def put(self):
        current_user = get_jwt_identity()

        if current_user != 'admin':
            return {'message': 'You are not authorized to access this resource'}, 403

        username = request.json.get('username')
        password = request.json.get('password')

        if username not in users:
            return {'message': 'User not found'}, 404
        users[username] = password
        return {'message': 'User updated successfully'}, 200
