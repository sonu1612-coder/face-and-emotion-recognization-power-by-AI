import tkinter as tk
from tkinter import messagebox
import cv2
from PIL import Image, ImageTk
import os
import uuid
import database

# Initialize cascade (ensure OpenCV is installed)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

class DataCollectionApp:
    def __init__(self, root, window_title="Face Mask & Emotion Data Collection"):
        self.root = root
        self.root.title(window_title)

        # Ensure database is initialized
        database.init_db()

        # OpenCV variables
        self.cap = cv2.VideoCapture(0)
        self.current_frame = None
        self.current_face = None # Holds cropped face image temporarily

        # UI Layout
        # Left Panel: Video Feed
        self.video_panel = tk.Label(self.root)
        self.video_panel.grid(row=0, column=0, padx=10, pady=10, rowspan=4)

        # Right Panel: Controls
        self.control_frame = tk.Frame(self.root)
        self.control_frame.grid(row=0, column=1, padx=10, pady=10, sticky="n")

        # Mask Label variable
        self.mask_var = tk.StringVar(value="No Mask")
        tk.Label(self.control_frame, text="Mask Status", font=("Arial", 14, "bold")).pack(pady=(0,5))
        tk.Radiobutton(self.control_frame, text="No Mask", variable=self.mask_var, value="No Mask").pack(anchor="w")
        tk.Radiobutton(self.control_frame, text="Mask", variable=self.mask_var, value="Mask").pack(anchor="w")

        # Emotion Label variable
        self.emotion_var = tk.StringVar(value="Neutral")
        tk.Label(self.control_frame, text="Emotion", font=("Arial", 14, "bold")).pack(pady=(20,5))
        emotions = ["Neutral", "Happy", "Sad", "Angry", "Surprise"]
        for em in emotions:
            tk.Radiobutton(self.control_frame, text=em, variable=self.emotion_var, value=em).pack(anchor="w")

        # Action Buttons
        self.btn_capture = tk.Button(self.control_frame, text="Capture Face", font=("Arial", 12), bg="lightblue", command=self.capture_and_save)
        self.btn_capture.pack(pady=30, fill="x")

        # Status Label
        self.status_label = tk.Label(self.control_frame, text="Capturing webcam stream...", fg="blue")
        self.status_label.pack(pady=10)

        # Start video loop
        self.update_video()

    def update_video(self):
        ret, frame = self.cap.read()
        if ret:
            # Resize frame for displaying smoothly
            frame = cv2.resize(frame, (640, 480))
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            self.current_frame = frame.copy()
            self.current_face = None
            
            # Draw bounding boxes
            for (x, y, w, h) in faces:
                cv2.rectangle(frame_rgb, (x, y), (x+w, y+h), (0, 255, 0), 2)
                # Keep the largest/last face as the active target
                # We crop from original BGR frame for saving
                self.current_face = self.current_frame[y:y+h, x:x+w]

            # Convert to PhotoImage for Tkinter
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_panel.imgtk = imgtk
            self.video_panel.configure(image=imgtk)

        # Call this function again after 20 milliseconds
        self.root.after(20, self.update_video)

    def capture_and_save(self):
        if self.current_face is None:
            messagebox.showwarning("Warning", "No face detected! Please ensure a face is reasonably visible in the frame.")
            return

        # Prepare paths and sizes
        dataset_dir = "dataset"
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
            
        # Resize all faces to 224x224 (good default for MobileNet)
        face_resized = cv2.resize(self.current_face, (224, 224))
        
        # Unique mapping
        filename = f"{uuid.uuid4().hex[:10]}.jpg"
        filepath = os.path.join(dataset_dir, filename)

        # Save to disk
        cv2.imwrite(filepath, face_resized)
        
        # Save to DB
        m_status = self.mask_var.get()
        e_status = self.emotion_var.get()
        database.insert_record(filepath, m_status, e_status)
        
        self.status_label.config(text=f"Saved: {m_status}, {e_status}", fg="green")

    def on_closing(self):
        self.cap.release()
        self.root.destroy()

if __name__ == '__main__':
    root = tk.Tk()
    app = DataCollectionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

