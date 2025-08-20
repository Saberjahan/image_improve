# Image Repair AI - Flask Backend

This directory contains the Flask backend application for the Image Repair AI project. It provides the API endpoints necessary for the React frontend to perform image processing tasks, including simulated AI detection, mask expansion, and inpainting.

---

## 📂 Project Structure

```
server/                     # Flask Backend Application
├── config/                 # Server configuration settings.
│   └── settings.py         # Defines CORS allowed origins.
├── routes/                 # Flask blueprints for API endpoints.
│   └── image_routes.py     # Defines the /api/image/process endpoint.
├── services/               # Business logic for AI operations (simulated).
│   └── ai_service.py       # Contains simulated AI processing functions.
├── venv/                   # Python virtual environment (ignored by Git).
├── app.py                  # Main Flask application entry point.
├── requirements.txt        # Python dependencies.
└── README.md               # This file.
```

---

## 🚀 Setup and Installation

Follow these steps to set up and run the Flask backend.

### 1. Navigate to the Server Directory

Open your terminal or command prompt and navigate to the `server` directory:

```bash
cd Image_Improve/server
```

### 2. Create and Activate a Virtual Environment

It's highly recommended to use a virtual environment to manage dependencies.

```bash
python -m venv venv
```

**Activate the virtual environment:**

* **On Windows:**
    ```bash
    .\venv\Scripts\activate
    ```
* **On macOS/Linux:**
    ```bash
    source venv/bin/activate
    ```

### 3. Install Python Dependencies

With your virtual environment activated, install the required Python packages using `pip`:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Backend

Once the setup is complete, you can start the Flask development server.

```bash
python -m flask run --port 3001 --debug
```

* The `--port 3001` argument specifies that the server will run on port 3001. This is important because the frontend expects to communicate with the backend on this port.
* The `--debug` flag enables debug mode, which provides detailed error messages and automatically reloads the server on code changes.

You should see output similar to this, indicating the server is running:

```
 * Debug mode: on
 * Running on [http://127.0.0.1:3001](http://127.0.0.1:3001)
```

The backend API will be accessible at `http://127.0.0.1:3001`.

---

## ⚙️ Configuration

* **`config/settings.py`**: This file contains important configurations, notably `CORS_ALLOWED_ORIGINS`. Ensure that the origin of your frontend application (e.g., `http://localhost:3000`) is listed here to prevent Cross-Origin Resource Sharing (CORS) issues.

---

## 💡 Simulated AI Services

The `services/ai_service.py` file contains placeholder functions that simulate AI image processing. In a real-world application, these functions would integrate with actual machine learning models or cloud AI APIs (e.g., Google Cloud Vision AI, OpenAI DALL-E, etc.).

* **`process_image_with_ai`**: Simulates various image manipulation tasks like detection, mask expansion, and inpainting.
* **`generate_keywords_from_description`**: Simulates generating relevant keywords from a text description, mimicking an LLM's capability.

These simulations allow the frontend to function end-to-end without requiring complex AI model setup locally.

---
