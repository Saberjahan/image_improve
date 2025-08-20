# server/config/settings.py

# Define allowed origins for CORS (Cross-Origin Resource Sharing).
# This is crucial for allowing your React frontend (localhost:3000)
# to make requests to your Flask backend (localhost:3001).
# Add other frontend origins here if your application is deployed elsewhere.
CORS_ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000"
    # Add any other frontend URLs if deployed
]

# You can add other backend-specific settings here, such as:
# - Database connection strings
# - API keys (though sensitive keys should ideally be in environment variables)
# - Debug mode settings (e.g., DEBUG = True/False)
# - Logging configurations
