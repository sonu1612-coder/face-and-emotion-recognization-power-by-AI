# CORE AI - Face Recognition & AI Assistant

Welcome to the **CORE AI** project! This is a real-time facial recognition, emotion analytics, mask detection, and psychological AI assistant dashboard built with Python, OpenCV, TensorFlow, and Flask.

---

## 🛠️ Prerequisites
Before running the project on a new computer, ensure you have the following installed:

1. **Git**
   - Download from: [git-scm.com](https://git-scm.com/downloads)
   - Allows you to clone the project from GitHub.
2. **Python 3.10 or 3.11** 
   - ⚠️ *Do NOT use Python 3.12+ as TensorFlow 2.16 may have compatibility issues.*
   - Download from: [python.org](https://www.python.org/downloads/)
   - **CRITICAL:** When installing Python, make sure to check the box that says **"Add Python to PATH"** at the very bottom of the installer window before clicking Install.

---

## 🚀 Step-by-Step Setup Guide

Follow these instructions perfectly to get the app running on any new machine.

### Step 1: Clone the Project from GitHub
Open your terminal (Command Prompt, PowerShell, or Git Bash) and run:
```bash
git clone https://github.com/sonu1612-coder/face-and-emotion-recognization-power-by-AI.git
cd face-and-emotion-recognization-power-by-AI
```

### Step 2: Create a Virtual Environment
We use a virtual environment to keep all project dependencies isolated safely.
Ensure you are inside the project folder you just cloned, then run:
```powershell
python -m venv venv
```

### Step 3: Install Required Dependencies
Once the virtual environment is built, you need to install the project packages. Run these installation commands carefully in your terminal to install everything required:

**Method 1: Install Manually (Recommended for new setups)**
Run this exact string to install all necessary Python libraries with their specific versions:
```powershell
.\venv\Scripts\pip.exe install opencv-contrib-python "tensorflow[and-cuda]==2.16.1" flask Pillow "numpy<2.0.0" google-generativeai python-dotenv protobuf==4.25.9 grpcio==1.62.1 grpcio-status==1.62.1
```

**Method 2: Install via Requirements file**
Alternatively, if your repository has the text file, run:
```powershell
.\venv\Scripts\pip.exe install -r requirements.txt
```
*(If you are on Mac/Linux, run: `source venv/bin/activate` followed by the `pip install` commands without the `.\venv\Scripts\` prefix)*

### Step 4: Setup the AI Chatbot (Gemini API)
For the CoreAI psychological assistant to converse with you, you need a free Google Gemini API key.
1. Get a free API key at: [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Inside the project folder, create a new text file and name it exactly `.env` (ensure it isn't saved as `.env.txt`).
3. Open it in Notepad or VS Code, and paste the following line with your new key:
```env
GEMINI_API_KEY="your_copied_api_key_here"
```

### Step 5: Download The AI Models (Required)
Inside the `models/` directory, ensure that `mask_model.h5` and `emotion_old.h5` are present. If they are ignored by GitHub (due to large file sizes), you must manually transfer these two model files to the new computer's `models/` folder.

### Step 6: Launch the Application
You are completely ready to go! Start the entire dashboard and server by running:
```powershell
.\venv\Scripts\python.exe main.py
```
*(Note: A browser window will automatically launch to the dashboard in the foreground. The AI system loading takes a few seconds to warm up!)*

---

## 👥 Meet the Team
- Daksh
- Aayush
- Athrva
- Zaid

---

## 🤖 AI Debugging Assistant Prompt
*Having trouble running the software? Copy the entire text in the box below and paste it into ChatGPT, Claude, or Gemini to give the AI the exact context needed to help you troubleshoot:*

```text
Act as an expert Python engineer and Computer Vision developer. I am trying to run a computer vision dashboard named "CoreAI" on my Windows machine. 

Here is the tech stack and architecture context of the application:
- Backend: Python 3.10/3.11 (not compatible with 3.12+), Flask
- AI & Vision: OpenCV (Haarcascade + LBPHFaceRecognizer), TensorFlow 2.16.1 (with PyTorch CUDA fallback routing), Google Generative AI (Gemini 1.5-flash)
- Front-End: HTML Canvas, CSS Glassmorphism, Vanilla JS
- Critical Dependencies: opencv-contrib-python, tensorflow[and-cuda]==2.16.1, flask, numpy<2.0.0, google-generativeai, protobuf==4.25.9, grpcio==1.62.1

The application connects to my webcam, runs a continuous generator thread in `app.py` to live-predict masks via Keras (`mask_model.h5`), emotions (`emotion_old.h5`), and identities, then streams the MJPEG result via a Flask `/video_feed` endpoint. Meanwhile, a chatbot endpoint (`/api/chat`) communicates with a virtual assistant via the Gemini API based on the currently detected emotion. Hardware initialization uses `cv2.CAP_DSHOW` and it launches locally.

I am experiencing an issue trying to run or install this project. Based on this complex threading and dependency architecture, please help me solve the following error/problem: 
[INSERT YOUR ERROR OR PROBLEM HERE]
```
