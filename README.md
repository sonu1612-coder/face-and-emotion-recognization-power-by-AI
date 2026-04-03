# Core AI - Face Recognition & AI Assistant

Welcome to our project! This is a real-time system for facial recognition, emotion detection, and mask detection. It also includes an AI chatbot assistant that interacts based on the detected emotion. The project runs on Python and uses OpenCV, TensorFlow, and Flask.

## Requirements

Make sure you have the following installed before running the project:
- **Git** to clone the repo.
- **Python 3.10 or 3.11** (Please don't use Python 3.12 or newer because TensorFlow 2.16 has compatibility issues with it). 
*Note: Make sure to check "Add Python to PATH" when installing.*

## How to Run the Project locally

Follow these steps to set up the project on your machine:

### 1. Clone the repository
Open your terminal or command prompt and run:
```bash
git clone https://github.com/sonu1612-coder/face-and-emotion-recognization-power-by-AI.git
cd face-and-emotion-recognization-power-by-AI
```

### 2. Set up a virtual environment
It's always better to use a virtual environment so you don't mess up your global Python packages.
```bash
python -m venv venv
```

### 3. Install packages
Install everything needed for the project. Make sure you are using the virtual environment you just created.
```bash
.\venv\Scripts\pip.exe install opencv-contrib-python "tensorflow[and-cuda]==2.16.1" flask Pillow "numpy<2.0.0" google-generativeai python-dotenv protobuf==4.25.9 grpcio==1.62.1 grpcio-status==1.62.1
```
Alternatively, if you want to use the requirements file:
```bash
.\venv\Scripts\pip.exe install -r requirements.txt
```
*(If you're using Mac or Linux, just activate the venv with `source venv/bin/activate` and use `pip install`)*

### 4. Setup Gemini API Key
The chatbot needs the Gemini API to work.
1. Get a free API key from Google AI Studio.
2. Create a new file in the project folder and name it exactly `.env`.
3. Put your API key in the `.env` file like this:
```env
GEMINI_API_KEY="your_api_key_here"
```

### 5. ML Models
Make sure you have `mask_model.h5` and `emotion_old.h5` inside the `models/` directory. Sometimes these files are too big for GitHub, so if they are missing, you will need to add them manually into the `models/` folder.

### 6. Start the App
To run the dashboard, execute the main script:
```bash
.\venv\Scripts\python.exe main.py
```
This will start the Flask server and automatically open the application in your browser. It might take a few seconds for everything to load the first time.

## Team Members
- Daksh
- Aayush
- Athrva
- Zaid
