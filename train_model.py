"""
CoreAI - Model Training
GPU-accelerated with mixed precision float16 and tf.data prefetching.
"""

import os
import pickle

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import database

# ── GPU Configuration ────────────────────────────────────────────────────────
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # float16 is ~2x faster on RTX 30xx (Ampere) with negligible accuracy loss
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        print(f"[GPU] {len(gpus)} GPU(s) ready — mixed precision float16 enabled.")
    except RuntimeError as e:
        print(f"[GPU] Config error: {e}")
else:
    print("[GPU] No GPU — training on CPU.")
# ─────────────────────────────────────────────────────────────────────────────

MASK_LABELS = {"No Mask": 0, "Mask": 1}
EMOTION_LABELS = {"Normal": 0, "Happy": 1, "Sad": 2, "Angry": 3, "Surprise": 4, "Disgusting": 5}
IMG_SIZE = 224

# Larger batch = better GPU utilization (RTX 3050 can handle 16 easily)
BATCH_SIZE = 16


def load_data():
    records = database.get_all_records()
    if not records:
        print("No records found. Collect data first.")
        return None, None, None

    images, mask_labels_list, emotion_labels_list = [], [], []
    print(f"Loading {len(records)} images from database...")

    for row in records:
        # Support both old schema (3 cols) and new schema (6 cols)
        if len(row) >= 6:
            _, img_path, mask_status, emotion, identity, timestamp = row
        else:
            img_path, mask_status, emotion = row[0], row[1], row[2]

        if not os.path.exists(img_path):
            continue
        img = cv2.imread(img_path)
        if img is None:
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
        mask_labels_list.append(MASK_LABELS.get(mask_status, 0))
        emotion_labels_list.append(EMOTION_LABELS.get(emotion, 0))

    if not images:
        print("No valid images found.")
        return None, None, None

    X = np.array(images, dtype=np.float32) / 255.0
    y_mask = np.array(mask_labels_list, dtype=np.int32)
    y_emotion = np.array(emotion_labels_list, dtype=np.int32)
    print(f"Loaded {len(X)} valid images.")
    return X, y_mask, y_emotion


def build_model(num_classes, activation='softmax'):
    """Build MobileNetV2-based classifier."""
    base_model = MobileNetV2(
        weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base_model.trainable = False  # Freeze pretrained layers

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    # Output must be float32 even with mixed precision
    predictions = Dense(num_classes, activation=activation, dtype='float32')(x)
    return Model(inputs=base_model.input, outputs=predictions)


def make_dataset(X, y, shuffle=True):
    """Create a tf.data dataset with prefetching for GPU pipeline."""
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(X), reshuffle_each_iteration=True)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


def train_mask_model(X, y):
    print("\n--- Training Mask Model ---")
    model = build_model(1, activation='sigmoid')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    val_split = 0.2 if len(X) > 10 else 0.0
    split_idx = int(len(X) * (1 - val_split))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    train_ds = make_dataset(X_train, y_train)
    callbacks = [
        EarlyStopping(patience=3, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(factor=0.5, patience=2, verbose=1)
    ]
    if val_split > 0:
        val_ds = make_dataset(X_val, y_val, shuffle=False)
        model.fit(train_ds, epochs=10, validation_data=val_ds, callbacks=callbacks)
    else:
        model.fit(train_ds, epochs=10)

    os.makedirs("models", exist_ok=True)
    model.save('models/mask_model.h5')
    print("Mask model saved → models/mask_model.h5")


def train_emotion_model(X, y):
    print("\n--- Training Emotion Model ---")
    model = build_model(len(EMOTION_LABELS), activation='softmax')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    val_split = 0.2 if len(X) > 10 else 0.0
    split_idx = int(len(X) * (1 - val_split))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    train_ds = make_dataset(X_train, y_train)
    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(factor=0.5, patience=3, verbose=1)
    ]
    if val_split > 0:
        val_ds = make_dataset(X_val, y_val, shuffle=False)
        model.fit(train_ds, epochs=30, validation_data=val_ds, callbacks=callbacks)
    else:
        model.fit(train_ds, epochs=20)

    os.makedirs("models", exist_ok=True)
    model.save('models/emotion_model.h5')
    print("Emotion model saved → models/emotion_model.h5")


if __name__ == '__main__':
    X, y_mask, y_emotion = load_data()
    if X is not None and len(X) > 0:
        os.makedirs("models", exist_ok=True)
        train_mask_model(X, y_mask)
        train_emotion_model(X, y_emotion)
        with open('models/label_map.pkl', 'wb') as f:
            pickle.dump({'mask': MASK_LABELS, 'emotion': EMOTION_LABELS}, f)
        print("\nTraining complete! Label mappings saved.")
    else:
        print("Not enough data to train. Collect more samples first.")
