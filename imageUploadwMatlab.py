import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.models import Sequential
import matplotlib.pyplot as plt

# — your base data directory —
base_data_dir = 'Tensorflow/pixyTrash/data'

# Parameters
IMG_HEIGHT, IMG_WIDTH = 150, 150
BATCH_SIZE = 32
EPOCHS = 10

# 2️⃣ Use image_dataset_from_directory to load & split data
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    base_data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode='int',                   # yields (images, integer labels)
    class_names=['Recyclables', 'Compost']
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    base_data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode='int',
    class_names=['Recyclables', 'Compost']
)

# Optional: prefetch for performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds   = val_ds.prefetch(buffer_size=AUTOTUNE)

# --- define a simple CNN ---
model = Sequential([
    Conv2D(32, (3,3), activation='relu',
           input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(2, activation='softmax')    # two classes
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# --- train ---
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds
)

# --- plot training history ---
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
