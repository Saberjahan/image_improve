# Image Repair AI

Welcome to the Image Repair AI project! This is a full-stack application designed to help you repair and restore damaged areas of images using a combination of manual drawing tools and AI-powered inpainting techniques.

## 🚀 Getting Started

You need to have Node.js and Python installed on your system. 

You have two options to get the application up and running on your local machine:

### Option 1: The Automated Way (Recommended)
For a faster setup, simply double-click the `start_project2.bat` file. This script will automatically:
- Download and install required libraries for both backend and frontend
- Set up virtual environments
- Start the application in your browser

### Option 2: The Manual Way
For a step-by-step guide on setting up the backend (Flask) and frontend (React) servers manually, refer to the `start_manual.txt` file. This document walks you through all the necessary commands, from installing dependencies to activating virtual environments.

## 🧠 Deep Learning Model

To use some of the deep-learning inpainting methods, you will need a pretrained model:
For Partial Convolution inpainting, you could use "https://drive.google.com/file/d/1sooo-BLSNRUGWG_AB-lxh7xHgJ2bS29a/view?usp=sharing" from https://github.com/tanimutomo/partialconv.
Put the model in ./../Image_Improve\model 

For now, the other deep learning methods are not set.

## ✨ Features

The application provides a comprehensive set of tools for targeted image restoration:

### Manual Tools
- **Precision Brush Tool:** Manually mark specific damaged areas that need repair
- **History Management:** Easily undo, redo, and reset your drawing and masking actions

### Intelligent Selection
- **Color Detection:** Automatically select and mask contiguous or global areas of specific colors
- **Smart Masking:** Useful for removing backgrounds or large patches of uniform color

### AI-Powered Restoration
- **Advanced Inpainting:** Apply classic and AI algorithms to fill masked areas intelligently

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

## ⚙️ Configuration Details

### CORS Setup (`config/settings.py`)
Ensures proper cross-origin communication between frontend and backend:
- **Frontend Origin:** `http://localhost:3000` (React dev server)
- **Backend Origin:** `http://127.0.0.1:3001` (Flask server)

## 📄 License

This project is open-source and available under the **MIT License**.

## 💡 Development Notes

- The backend uses simulated AI processing during development

