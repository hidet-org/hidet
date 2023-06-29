import os

from flask import Flask, send_from_directory
from flask_jwt_extended import JWTManager
from flask_restful import Api

from resources import CompilationResource, AuthResource, UserResource

app = Flask(__name__)
app.config['JWT_SECRET_KEY'] = os.urandom(32)
api = Api(app)
jwt = JWTManager(app)

api.add_resource(CompilationResource, '/compile')
api.add_resource(AuthResource, '/auth')
api.add_resource(UserResource, '/user')


@app.route('/download/<string:filename>')
def download(filename):
    import hidet
    cache_dir = hidet.utils.cache_dir('compilefarm')
    path = os.path.join(cache_dir, filename)
    if os.path.exists(path):
        return send_from_directory(cache_dir, filename, as_attachment=True)
    else:
        return 'File not found', 404


if __name__ == '__main__':
    app.run(debug=True)
