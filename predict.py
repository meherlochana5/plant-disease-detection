import tensorflow as tf
import numpy as np
import json
import os
import gdown
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input

MODEL_PATH = "models/final_model.h5"
FILE_ID = "1gM7Nn9ukPp-XahWNP9XBNjXYr4rYW8KV"

# Create models folder
os.makedirs("models", exist_ok=True)

# Download model if missing
if not os.path.exists(MODEL_PATH):
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

with open("class_indices.json", "r") as f:
    labels = json.load(f)

classes = {v: k for k, v in labels.items()}

def is_leaf(img):
    arr = np.array(img.resize((224, 224)))
    if arr.ndim != 3:
        return False

    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]

    green_pixels = ((g > r) & (g > b)).mean()
    return green_pixels > 0.20

def predict_image(path):
    img = Image.open(path).convert("RGB")

    if not is_leaf(img):
        return "Invalid Image! Upload leaf image only.", 0

    img = img.resize((224, 224))
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    pred = model.predict(img)[0]
    idx = np.argmax(pred)
    confidence = round(float(np.max(pred)) * 100, 2)

    result = classes[idx]

    if "healthy" in result.lower():
        result += " (Healthy Leaf)"
    else:
        result += " (Disease Detected)"

    return result, confidence
