from flask import Flask
from app.routes import api

def create_app():
    app = Flask(__name__)
    app.register_blueprint(api)
    return app

app = create_app()

if __name__ == "__main__":
    app.run(debug=True)