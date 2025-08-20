# Image Repair AI

Welcome to the Image Repair AI project! This is a full-stack application designed to help you repair and restore damaged areas of images using a combination of manual drawing tools and AI-powered inpainting techniques.

## 🚀 Getting Started

You have two options to get the application up and running on your local machine:

### Option 1: The Automated Way (Recommended)
For a faster setup, simply double-click the `start_project2.bat` file. This script will automatically:
- Download and install required libraries for both backend and frontend
- Set up virtual environments
- Start the application in your browser

### Option 2: The Manual Way
For a step-by-step guide on setting up the backend (Flask) and frontend (React) servers manually, refer to the `start_manual.txt` file. This document walks you through all the necessary commands, from installing dependencies to activating virtual environments.

## 🧠 Deep Learning Model

To use the deep-learning inpainting methods, you will need a pretrained model:

- **Model File:** `gated_conv_model.keras`
- **Type:** Partial Convolution Inpainting model
- **Format:** Keras model that loads directly into the Python backend
- **Purpose:** Performs AI-based intelligent image area restoration

## ✨ Features

The application provides a comprehensive set of tools for targeted image restoration:

### Manual Tools
- **Precision Brush Tool:** Manually mark specific damaged areas that need repair
- **History Management:** Easily undo, redo, and reset your drawing and masking actions

### Intelligent Selection
- **Color Detection:** Automatically select and mask contiguous or global areas of specific colors
- **Smart Masking:** Useful for removing backgrounds or large patches of uniform color

### AI-Powered Restoration
- **Advanced Inpainting:** Apply cutting-edge AI algorithms to intelligently fill masked areas
- **Seamless Integration:** Produces natural-looking repairs that blend perfectly with surrounding content

## 📁 Project Structure

```
.
├── server/                     # Flask Backend Application
│   ├── config/                 # Server configuration settings
│   │   └── settings.py         # CORS and server configurations
│   ├── routes/                 # Flask API endpoints
│   │   └── image_routes.py     # Image processing endpoints
│   ├── services/               # AI processing business logic
│   │   └── ai_service.py       # Core AI inpainting functions
│   ├── venv/                   # Python virtual environment
│   ├── app.py                  # Flask application entry point
│   └── requirements.txt        # Python dependencies
├── client/                     # React Frontend Application
│   ├── public/                 # Static frontend assets
│   ├── src/                    # React source code
│   │   ├── components/         # Reusable UI components
│   │   ├── features/           # Feature-specific modules
│   │   ├── hooks/              # Custom React hooks
│   │   └── App.js              # Main React application
│   ├── package.json            # Node.js dependencies
│   └── package-lock.json       # Dependency lock file
├── gated_conv_model.keras      # Pretrained inpainting model
├── start_manual.txt            # Manual setup guide
├── start_project2.bat          # Automated setup script
└── README.md                   # Project documentation
```

---

# Flask Backend Documentation

## 📂 Backend Architecture

The Flask backend serves as the core processing engine for image restoration operations:

## 🛠 Setup and Installation

### 1. Navigate to Backend Directory
```bash
cd Image_Improve/server
```

### 2. Create Virtual Environment
```bash
python -m venv venv
```

### 3. Activate Virtual Environment

**Windows:**
```bash
.\venv\Scripts\activate
```

**macOS/Linux:**
```bash
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

## ▶️ Running the Backend Server

Start the Flask development server with debug mode enabled:

```bash
python -m flask run --port 3001 --debug
```

**Server Configuration:**
- **Port:** 3001 (required for frontend communication)
- **Debug Mode:** Enabled for development
- **Auto-reload:** Automatic server restart on code changes
- **Access URL:** `http://127.0.0.1:3001`

Expected output:
```
 * Debug mode: on
 * Running on http://127.0.0.1:3001
 * Restarting with stat
 * Debugger is active!
```

## ⚙️ Configuration Details

### CORS Setup (`config/settings.py`)
Ensures proper cross-origin communication between frontend and backend:
- **Frontend Origin:** `http://localhost:3000` (React dev server)
- **Backend Origin:** `http://127.0.0.1:3001` (Flask server)
- **Security:** Prevents unauthorized cross-origin requests

## 🤖 AI Service Implementation

The `services/ai_service.py` module contains the core image processing logic:

### Key Functions
- **`process_image_with_ai`:** Main inpainting processing pipeline
- **`generate_keywords_from_description`:** Text-to-prompt generation for guided inpainting
- **Model Integration:** Direct integration with `gated_conv_model.keras`

### Processing Pipeline
1. **Image Input:** Receives damaged image and mask data
2. **Preprocessing:** Normalizes and prepares data for AI model
3. **Inpainting:** Applies Partial Convolution algorithms
4. **Post-processing:** Refines output for seamless integration
5. **Return:** Delivers restored image to frontend

## 🔧 API Endpoints

### `/api/image/process` (POST)
**Purpose:** Process image restoration requests

**Input:**
- Base64 encoded image data
- Mask coordinates/regions
- Processing parameters

**Output:**
- Restored image data
- Processing status
- Metadata about restoration

## 📄 License

This project is open-source and available under the **MIT License**.

---

## 💡 Development Notes

- The backend uses simulated AI processing during development
- Production deployment requires actual model loading and GPU optimization
- CORS configuration must be updated for production domains
- Virtual environment isolation ensures consistent dependency management
