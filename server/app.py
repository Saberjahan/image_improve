# server/app.py

from flask import Flask, jsonify
from flask_cors import CORS
from routes.image_routes import image_bp # Import the blueprint
from config.settings import CORS_ALLOWED_ORIGINS # Import CORS settings

app = Flask(__name__)

# Configure CORS to allow requests from the specified origins
# This is crucial for your React frontend to communicate with this Flask backend.
CORS(app, resources={r"/api/*": {"origins": CORS_ALLOWED_ORIGINS}})

# Register the image blueprint
app.register_blueprint(image_bp, url_prefix='/api')

@app.route('/')
def home():
    """
    A simple home route to confirm the server is running.
    """
    return jsonify({"message": "Image Repair AI Backend is running!"})

if __name__ == '__main__':
    # Run the Flask application
    # In a production environment, you would use a production-ready WSGI server
    # like Gunicorn or uWSGI. For development, flask run is sufficient.
    app.run(port=3001, debug=True)
