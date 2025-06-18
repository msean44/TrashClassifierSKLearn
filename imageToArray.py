import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from keras.preprocessing.image import img_to_array
from keras.applications.mobilenet_v2 import preprocess_input
from keras.models import load_model

MODEL_PATH = 'trash_classifier_model.h5'
CLASS_LABELS = ['recyclables', 'food/compost', 'paper', 'waste']

#Pre-trained model
model = load_model(MODEL_PATH)

#File-picker function
def pick_image_file():

    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(
        title = "Select an image",
        filetypes=[("JPEG files", "*.jpg;*.jpeg"), ("PNG files", "*.png"), ("All files", "*.*")]
    )

    return file_path

#Pick file
img_path = pick_image_file()

if not img_path:
    print("No file selected")
    exit()

orig = cv2.imread(img_path)
if orig is None:
    print("Failed to read image, please check the file")
    exit()

cv2.imshow("Uploaded image", orig)
cv2.waitKey(0)
cv2.destroyAllWindows

#Resize and preprocess to 224x224
frame_resized = cv2.resize(orig, (224,224))
image = img_to_array(frame_resized)
image = preprocess_input(image)
image = np.expand_dims(image, axis = 0)

#Predict model
preds = model.predict(image)
idx = np.argmax(preds[0])
label = CLASS_LABELS[idx]
confidence = preds[0][idx]
print(f"Prediction: {label} ({confidence*100:.2f}%)")