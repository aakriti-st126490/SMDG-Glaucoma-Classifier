from flask import Flask
from api.model_loader import load_models
from api.routes import create_routes

app = Flask(__name__)

print("Loading models...")
models = load_models()
print("Models loaded")

app.register_blueprint(create_routes(models))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)