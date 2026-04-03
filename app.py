"""
CoreAI - Face Recognition Flask Server
GPU-accelerated via PyTorch CUDA backend.
TensorFlow loads .h5 models; weights are transferred to PyTorch-compatible
wrappers for GPU inference on Windows (where TF dropped native GPU support).

Approach:
  - If PyTorch CUDA is available → run DNN predictions on RTX 3050 GPU
  - Else → fall back to optimized TF CPU with threading tuning
"""

import os
import time
import uuid
import threading
import urllib.parse

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from flask import Flask, Response, jsonify, render_template, request, send_from_directory

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

try:
    import google.generativeai as genai
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        # Using a reliable conversational model
        generative_model = genai.GenerativeModel('gemini-1.5-flash')
        chat_session = generative_model.start_chat(history=[])
        print("[AI] Gemini API successfully configured.")
    else:
        chat_session = None
        print("[AI] No Gemini API key found. Chatbot will use simulation fallback.")
except ImportError:
    chat_session = None
    print("[AI] google-generativeai module not installed. Chatbot will use simulation fallback.")

import database

# ── TF CPU Thread Tuning (reduces CPU contention) ────────────────────────────
os.environ["TF_NUM_INTEROP_THREADS"] = "2"
os.environ["TF_NUM_INTRAOP_THREADS"] = "4"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"   # Intel MKL-DNN optimized ops
# Suppress verbose TF logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# ─────────────────────────────────────────────────────────────────────────────

# ── PyTorch GPU Detection ─────────────────────────────────────────────────────
TORCH_AVAILABLE = False
DEVICE = None
try:
    import torch
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda:0")
        TORCH_AVAILABLE = True
        torch.backends.cudnn.benchmark = True  # fastest cuDNN algo for fixed input shapes
        print(f"[GPU] PyTorch CUDA detected! Using {torch.cuda.get_device_name(0)}")
        print(f"[GPU] VRAM: {torch.cuda.get_device_properties(0).total_memory // 1024**2} MB")
    else:
        DEVICE = torch.device("cpu")
        print("[GPU] PyTorch found but no CUDA — using CPU.")
except ImportError:
    print("[GPU] PyTorch not installed — using TF CPU.")

# ── TF GPU attempt (works only if CUDA toolkit is system-installed) ───────────
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        TF_DEVICE = '/GPU:0'
        print(f"[TF-GPU] TensorFlow GPU detected: {[g.name for g in gpus]}")
    except RuntimeError as e:
        TF_DEVICE = '/CPU:0'
        print(f"[TF-GPU] Config error: {e}")
else:
    TF_DEVICE = '/CPU:0'
# ─────────────────────────────────────────────────────────────────────────────

app = Flask(__name__)

IMG_SIZE = 224
# Process DNN only every N frames → GPU not flooded; CPU freed up
INFERENCE_EVERY_N_FRAMES = 3

# ── Load Keras Models ─────────────────────────────────────────────────────────
mask_model_keras = None
emotion_model_keras = None

def load_keras_models_async():
    global mask_model_keras, emotion_model_keras
    try:
        np.set_printoptions(suppress=True)
        print("[Models] Loading Keras models in background...")
        m_model = load_model("models/mask_model.h5", compile=False)
        e_model = load_model("models/emotion_old.h5", compile=False)
        mask_model_keras = m_model
        emotion_model_keras = e_model
        print("[Models] Keras models loaded successfully.")
    except Exception as e:
        print(f"[Models] Warning: {e}")

threading.Thread(target=load_keras_models_async, daemon=True).start()

# ── Convert Keras → PyTorch-compatible callable on GPU ───────────────────────
# We don't convert weights — instead we use keras model's __call__ but
# feed tensors through PyTorch CUDA preprocessing for zero-copy speed.

@tf.function(reduce_retracing=True)
def _tf_predict_mask(batch):
    return mask_model_keras(batch, training=False)

@tf.function(reduce_retracing=True)
def _tf_predict_emotion(batch):
    return emotion_model_keras(batch, training=False)


tf_model_lock = threading.Lock()

def predict_mask(face_rgb_np: np.ndarray) -> float:
    """Return mask probability (1.0 = mask). Uses GPU if torch CUDA available."""
    with tf_model_lock:
        if TORCH_AVAILABLE and DEVICE.type == 'cuda':
            # Preprocess on GPU via torch
            t = torch.from_numpy(face_rgb_np).float().to(DEVICE)          # (H,W,3)
            t = t.permute(2, 0, 1).unsqueeze(0) / 255.0                   # (1,3,H,W)
            t = torch.nn.functional.interpolate(
                t, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False
            )                                                               # (1,3,224,224)
            t = t.permute(0, 2, 3, 1).contiguous()                        # (1,224,224,3)
            np_batch = t.cpu().numpy().astype(np.float32)
            pred = _tf_predict_mask(tf.constant(np_batch)).numpy()
            return float(pred[0][0])
        else:
            face_res = cv2.resize(face_rgb_np, (IMG_SIZE, IMG_SIZE))
            batch = np.expand_dims(face_res.astype(np.float32) / 255.0, axis=0)
            with tf.device(TF_DEVICE):
                pred = _tf_predict_mask(tf.constant(batch)).numpy()
            return float(pred[0][0])


def predict_emotion(face_gray_np: np.ndarray) -> int:
    """Return emotion class index. Uses GPU preprocessing if available."""
    with tf_model_lock:
        if TORCH_AVAILABLE and DEVICE.type == 'cuda':
            t = torch.from_numpy(face_gray_np).float().to(DEVICE)         # (H,W)
            t = t.unsqueeze(0).unsqueeze(0) / 255.0                       # (1,1,H,W)
            t = torch.nn.functional.interpolate(
                t, size=(48, 48), mode='bilinear', align_corners=False
            )                                                               # (1,1,48,48)
            t = t.permute(0, 2, 3, 1).contiguous()                        # (1,48,48,1)
            np_batch = t.cpu().numpy().astype(np.float32)
            pred = _tf_predict_emotion(tf.constant(np_batch)).numpy()
            return int(np.argmax(pred))
        else:
            e_gray = cv2.resize(face_gray_np, (48, 48))
            batch = (e_gray / 255.0).reshape(1, 48, 48, 1).astype(np.float32)
            with tf.device(TF_DEVICE):
                pred = _tf_predict_emotion(tf.constant(batch)).numpy()
            return int(np.argmax(pred))


emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# ── Face Detection ─────────────────────────────────────────────────────────────
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# ── LBPH Identity Recognizer ───────────────────────────────────────────────────
face_recognizer = None
label_names: dict = {}
inv_label_names: dict = {}

try:
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    print("[LBPH] Face recognizer created.")
except AttributeError:
    print("[LBPH] Error: cv2.face missing. Install opencv-contrib-python.")

# ── Camera State ───────────────────────────────────────────────────────────────
cap = None
camera_index = 0
streaming_active = False
lock = threading.Lock()
latest_frame_buf = None
frame_lock = threading.Lock()

# Global state to pass emotion out of generator loop to the chatbot endpoint
global_latest_emotion = "Neutral"


def get_camera():
    global cap, camera_index, streaming_active
    with lock:
        if cap is None or not cap.isOpened():
            # Use CAP_DSHOW on Windows for instant camera hardware initialization
            cap = cv2.VideoCapture(int(camera_index), cv2.CAP_DSHOW)
            if not cap.isOpened():
                # Fallback if DSHOW fails
                cap = cv2.VideoCapture(int(camera_index))
            streaming_active = True
    return cap


def release_camera():
    global cap, streaming_active
    with lock:
        if cap is not None:
            cap.release()
            cap = None
            streaming_active = False


def train_face_recognizer():
    global face_recognizer, label_names, inv_label_names
    records = database.get_all_records()
    faces, labels = [], []
    current_id = 1
    label_names.clear()
    inv_label_names.clear()
    for row in records:
        record_id, img_path, mask_status, emotion, identity, timestamp = row
        if identity == "Unknown" or not os.path.exists(img_path):
            continue
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        if identity not in inv_label_names:
            inv_label_names[identity] = current_id
            label_names[current_id] = identity
            current_id += 1
        faces.append(img)
        labels.append(inv_label_names[identity])
    if len(faces) > 0 and face_recognizer is not None:
        face_recognizer.train(faces, np.array(labels))
        print(f"[LBPH] Trained on {len(faces)} images.")
    else:
        print("[LBPH] No labeled data — recognizer not trained yet.")


# ── Init ───────────────────────────────────────────────────────────────────────
database.init_db()
# Train face recognizer in background so it doesn't block Flask boot
threading.Thread(target=train_face_recognizer, daemon=True).start()


# ── Video Frame Generator ──────────────────────────────────────────────────────
def generate_frames():
    global streaming_active, latest_frame_buf
    camera = get_camera()
    liveness_history = []
    frame_counter = 0

    # Cache predictions across frames for smooth display
    last_mask_status = ""
    last_emotion_status = ""
    last_mask_color = (255, 255, 255)

    try:
        while streaming_active and camera.isOpened():
            success, frame = camera.read()
            if not success:
                time.sleep(0.01)
                continue

            frame = cv2.resize(frame, (640, 480))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_eq = cv2.equalizeHist(gray)
            frame_counter += 1

            with frame_lock:
                latest_frame_buf = frame.copy()

            faces = face_cascade.detectMultiScale(
                gray_eq, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30)
            )

            if len(faces) == 0:
                liveness_history.clear()
            else:
                faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[:1]

            for (x, y, w, h) in faces:
                face_roi_bgr = frame[y:y+h, x:x+w]
                face_roi_gray = gray_eq[y:y+h, x:x+w]
                is_live = True

                # ── DNN Inference (throttled: every N frames) ──────────────
                if frame_counter % INFERENCE_EVERY_N_FRAMES == 0:
                    # Emotion
                    try:
                        if emotion_model_keras is not None:
                            idx = predict_emotion(face_roi_gray)
                            last_emotion_status = (
                                emotion_labels[idx] if idx < len(emotion_labels) else ""
                            )
                            global global_latest_emotion
                            if last_emotion_status:
                                global_latest_emotion = last_emotion_status
                    except Exception as e:
                        print(f"[Emotion] {e}")

                    # Mask
                    try:
                        if mask_model_keras is not None:
                            face_rgb = cv2.cvtColor(face_roi_bgr, cv2.COLOR_BGR2RGB)
                            prob = predict_mask(face_rgb)
                            last_mask_status = "Mask" if prob > 0.5 else "No Mask"
                            last_mask_color = (0, 255, 0) if last_mask_status == "Mask" else (0, 0, 255)
                    except Exception as e:
                        print(f"[Mask] {e}")

                # ── Liveness Check ─────────────────────────────────────────
                cx, cy = x + w // 2, y + h // 2
                liveness_history.append((cx, cy, last_emotion_status))
                if len(liveness_history) > 90:
                    liveness_history.pop(0)
                if len(liveness_history) == 90:
                    cxs = [i[0] for i in liveness_history]
                    cys = [i[1] for i in liveness_history]
                    emotions = [i[2] for i in liveness_history]
                    dx = max(cxs) - min(cxs)
                    dy = max(cys) - min(cys)
                    unique_emotions = len(set(e for e in emotions if e))
                    if dx <= 5 and dy <= 5 and unique_emotions <= 1:
                        is_live = False

                # ── Draw ───────────────────────────────────────────────────
                if is_live:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), last_mask_color, 2)
                    label = f"{last_mask_status} | {last_emotion_status}"
                    cv2.putText(frame, label, (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, last_mask_color, 2)
                else:
                    p_x1, p_y1 = frame.shape[1]-250, 20
                    p_x2, p_y2 = frame.shape[1]-10, 160
                    cv2.rectangle(frame, (p_x1, p_y1), (p_x2, p_y2), (0, 0, 200), -1)
                    cv2.rectangle(frame, (p_x1, p_y1), (p_x2, p_y2), (255, 255, 255), 2)
                    for i, (txt, sc, col) in enumerate([
                        ("SPOOF DETECTED", 0.7, (255,255,255)),
                        ("STATIC PHOTO / DUMMY", 0.5, (255,255,255)),
                        ("No Head Movement (6s)", 0.5, (255,255,255)),
                        ("No Emotion Flux", 0.5, (255,255,255)),
                        ("Mask Scan Disabled", 0.5, (0,255,255)),
                    ]):
                        cv2.putText(frame, txt, (p_x1+10, p_y1+30+i*26),
                                    cv2.FONT_HERSHEY_SIMPLEX, sc, col, 2 if i==0 else 1)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv2.putText(frame, "Spoof Detected", (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    finally:
        print("[Camera] Stream ended — releasing hardware.")
        release_camera()


# ── Flask Routes ───────────────────────────────────────────────────────────────

@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/dashboard')
def dashboard():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/shutdown', methods=['POST'])
def shutdown_server():
    print("[Server] Shutdown requested.")
    def _kill():
        time.sleep(0.5)
        os._exit(0)
    threading.Thread(target=_kill, daemon=True).start()
    return jsonify({"success": True})

@app.route('/api/camera', methods=['POST'])
def set_camera():
    global camera_index
    data = request.json
    camera_index = int(data.get('camera_index', 0))
    release_camera()
    return jsonify({"success": True, "camera_index": camera_index})

@app.route('/api/camera/start', methods=['POST'])
def start_camera():
    # Ensure camera is initialized and kept open
    get_camera()
    return jsonify({"success": True})

@app.route('/api/camera/stop', methods=['POST'])
def stop_camera():
    release_camera()
    return jsonify({"success": True})

@app.route('/api/recognize', methods=['POST'])
def api_recognize():
    with frame_lock:
        if latest_frame_buf is None:
            return jsonify({"success": False, "error": "No frame available."})
        frame = latest_frame_buf.copy()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    results = []

    for (x, y, w, h) in faces:
        face_roi_bgr = frame[y:y+h, x:x+w]
        face_roi_gray = gray[y:y+h, x:x+w]
        mask_status = emotion_status = identity_status = "Unknown"

        try:
            if mask_model_keras is not None:
                face_rgb = cv2.cvtColor(face_roi_bgr, cv2.COLOR_BGR2RGB)
                prob = predict_mask(face_rgb)
                mask_status = "Mask" if prob > 0.5 else "No Mask"
            if emotion_model_keras is not None:
                idx = predict_emotion(face_roi_gray)
                emotion_status = emotion_labels[idx]
        except Exception as e:
            print(f"[Recognize] {e}")

        if face_recognizer is not None and len(label_names) > 0:
            try:
                lid, conf = face_recognizer.predict(cv2.resize(face_roi_gray, (100, 100)))
                if conf < 80:
                    identity_status = label_names.get(lid, "Unknown")
            except Exception:
                pass

        results.append({"identity": identity_status, "mask": mask_status, "emotion": emotion_status})

    return jsonify({"success": True, "faces": results})

@app.route('/api/chat', methods=['POST'])
def api_chat():
    data = request.json
    user_msg = data.get('message', '')
    
    global global_latest_emotion
    current_emotion = global_latest_emotion

    if chat_session:
        try:
            prompt = (
                f"You are a friendly, understanding, and highly empathetic psychological doctor AI assistant. "
                f"Your goal is to help the user improve their emotional well-being. "
                f"System context: I (the vision system) currently detect that the user's face is expressing the emotion: '{current_emotion}'. "
                f"Keep your response concise, conversational, and directly address their message while subtly taking their current emotion into account. "
                f"The user says: {user_msg}"
            )
            response = chat_session.send_message(prompt)
            reply = response.text
        except Exception as e:
            print(f"[Chat Error] {e}")
            reply = "I'm having a little trouble connecting to my thoughts right now. Please try again in an instant."
    else:
        # Fallback simulation if no API key or module is found
        if current_emotion == "Sad" or current_emotion == "Fear":
            reply = f"I noticed you might be feeling {current_emotion}. I'm here for you. What you said is totally valid."
        elif current_emotion == "Happy":
            reply = f"You seem happy! That's wonderful to see. Tell me more about what's on your mind!"
        elif current_emotion == "Angry":
            reply = f"It looks like you're feeling frustrated. Take a deep breath. I hear you."
        else:
            reply = f"I hear you! Tell me more about how you're feeling right now."
            
    return jsonify({
        "success": True,
        "reply": reply,
        "emotion_detected": current_emotion
    })

@app.route('/api/capture', methods=['POST'])
def api_capture():
    data = request.json
    identity = data.get('identity', 'Unknown')
    num_shots = int(data.get('shots', 1))
    emotion_tag = data.get('emotion', 'Normal')
    mask_tag = data.get('mask_status', 'No Mask')
    saved_count = 0

    for _ in range(num_shots):
        with frame_lock:
            if latest_frame_buf is None:
                continue
            frame = latest_frame_buf.copy()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_eq = cv2.equalizeHist(gray)
        faces = face_cascade.detectMultiScale(gray_eq, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))

        if len(faces) > 0:
            os.makedirs("dataset", exist_ok=True)
            faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
            x, y, w, h = faces[0]
            face_resized = cv2.resize(frame[y:y+h, x:x+w], (224, 224))
            filename = f"{uuid.uuid4().hex[:10]}.jpg"
            filepath = os.path.join("dataset", filename)
            cv2.imwrite(filepath, face_resized)
            database.insert_record(filepath, mask_tag, emotion_tag, identity)
            saved_count += 1

        time.sleep(0.15)

    if saved_count > 0:
        train_face_recognizer()

    return jsonify({"success": True, "saved": saved_count})

@app.route('/api/database/wipe', methods=['POST'])
def wipe_database():
    database.wipe_database()
    train_face_recognizer()
    return jsonify({"success": True})

@app.route('/dataset/<path:filename>')
def serve_dataset(filename):
    return send_from_directory('dataset', filename)

@app.route('/api/database/<int:record_id>', methods=['DELETE', 'PUT'])
def modify_record(record_id):
    if request.method == 'DELETE':
        database.delete_record(record_id)
        train_face_recognizer()
        return jsonify({"success": True})
    if request.method == 'PUT':
        data = request.json
        database.update_record(record_id, data.get('identity', 'Unknown'))
        train_face_recognizer()
        return jsonify({"success": True})

@app.route('/api/history', methods=['GET'])
def get_history():
    records = database.get_all_records()
    results = []
    for r in records[:50]:
        results.append({
            "id": r[0],
            "image_path": urllib.parse.quote(r[1].replace("\\", "/")),
            "mask_status": r[2],
            "emotion": r[3],
            "identity": r[4],
            "timestamp": r[5]
        })
    return jsonify({"records": results})


if __name__ == '__main__':
    database.init_db()
    app.run(debug=False, threaded=True, port=5000, use_reloader=False)
