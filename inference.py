"""Real-Time Inference App for Face Mask & Emotion Recognition (GPU-accelerated)."""

import os
import typing
import tkinter as tk
from tkinter import messagebox

import cv2  # type: ignore
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model  # type: ignore
from PIL import Image, ImageTk

cv2 = typing.cast(typing.Any, cv2)

# ── GPU Configuration ────────────────────────────────────────────────────────
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        DEVICE = '/GPU:0'
        print(f"[GPU] Using GPU for inference: {[g.name for g in gpus]}")
    except RuntimeError as e:
        print(f"[GPU] Config error: {e}")
        DEVICE = '/CPU:0'
else:
    DEVICE = '/CPU:0'
    print("[GPU] No GPU detected — using CPU.")
# ─────────────────────────────────────────────────────────────────────────────

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

IMG_SIZE = 224
INFERENCE_EVERY_N_FRAMES = 3  # Throttle: run DNN every 3rd frame


class InferenceApp:
    """Main Application for Real-Time Face Mask and Emotion Inference."""

    def __init__(self, root, window_title="Real-Time Inference — GPU Accelerated"):
        self.root = root
        self.root.title(window_title)
        self.mask_model = None
        self.emotion_model = None
        self._frame_counter = 0
        self._last_mask = ""
        self._last_emotion = ""
        self._last_color = (255, 255, 255)
        self.emotion_labels = ["Neutral", "Happy", "Sad", "Angry", "Surprise"]

        if not os.path.exists("models/mask_model.h5") or not os.path.exists("models/emotion_model.h5"):
            messagebox.showerror("Error", "Models not found! Train models first.")
            self.root.destroy()
            return

        status = tk.Label(self.root, text="Loading models…", font=("Arial", 12))
        status.pack(pady=20)
        self.root.update()

        try:
            self.mask_model = load_model("models/mask_model.h5", compile=False)
            self.emotion_model = load_model("models/emotion_model.h5", compile=False)
        except Exception as e:
            messagebox.showerror("Model Load Error", str(e))
            self.root.destroy()
            return

        status.pack_forget()

        self.video_panel = tk.Label(self.root)
        self.video_panel.pack(padx=10, pady=10)

        self.cap = cv2.VideoCapture(0)
        self.update_video()

    @tf.function(reduce_retracing=True)
    def _infer_mask(self, batch):
        return self.mask_model(batch, training=False)

    @tf.function(reduce_retracing=True)
    def _infer_emotion(self, batch):
        return self.emotion_model(batch, training=False)

    def update_video(self):
        """Update the video feed with frames from the webcam."""
        ret, frame = self.cap.read()
        if ret:
            self._frame_counter += 1
            frame = cv2.resize(frame, (640, 480))
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_eq = cv2.equalizeHist(gray)

            faces = face_cascade.detectMultiScale(gray_eq, 1.1, 4, minSize=(30, 30))

            for (x, y, w, h) in faces:
                face_roi = frame[y:y+h, x:x+w]

                # Only run DNN every N frames
                if self._frame_counter % INFERENCE_EVERY_N_FRAMES == 0:
                    try:
                        if self.mask_model is not None:
                            face_res = cv2.resize(face_roi, (IMG_SIZE, IMG_SIZE))
                            face_rgb_small = cv2.cvtColor(face_res, cv2.COLOR_BGR2RGB)
                            m_batch = np.expand_dims(
                                np.array(face_rgb_small, dtype=np.float32) / 255.0, axis=0
                            )
                            with tf.device(DEVICE):
                                m_pred = self._infer_mask(tf.constant(m_batch)).numpy()
                            self._last_mask = "Mask" if m_pred[0][0] > 0.5 else "No Mask"
                            self._last_color = (0, 255, 0) if self._last_mask == "Mask" else (0, 0, 255)
                    except Exception as e:
                        print(f"[Mask] {e}")

                    try:
                        if self.emotion_model is not None:
                            face_g = cv2.resize(gray_eq[y:y+h, x:x+w], (48, 48))
                            e_batch = (face_g / 255.0).reshape(1, 48, 48, 1).astype(np.float32)
                            with tf.device(DEVICE):
                                e_pred = self._infer_emotion(tf.constant(e_batch)).numpy()
                            idx = int(np.argmax(e_pred))
                            self._last_emotion = (
                                self.emotion_labels[idx] if idx < len(self.emotion_labels) else ""
                            )
                    except Exception as e:
                        print(f"[Emotion] {e}")

                cv2.rectangle(frame_rgb, (x, y), (x+w, y+h), self._last_color, 2)
                label = f"{self._last_mask} — {self._last_emotion}"
                cv2.putText(frame_rgb, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, self._last_color, 2)

            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_panel.imgtk = imgtk  # type: ignore
            self.video_panel.configure(image=imgtk)

        self.root.after(20, self.update_video)

    def on_closing(self):
        """Release the capture device and destroy the window."""
        self.cap.release()
        self.root.destroy()


if __name__ == '__main__':
    tk_root = tk.Tk()
    app = InferenceApp(tk_root)
    tk_root.protocol("WM_DELETE_WINDOW", app.on_closing)
    tk_root.mainloop()
