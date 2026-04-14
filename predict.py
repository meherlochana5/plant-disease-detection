import tensorflow as tf
import numpy as np
import json
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input

model = tf.keras.models.load_model("models/final_model.h5")

with open("class_indices.json", "r") as f:
    labels = json.load(f)

classes = {v:k for k,v in labels.items()}

def is_leaf(img):
    arr = np.array(img.resize((224,224)))
    if arr.ndim != 3:
        return False

    r = arr[:,:,0]
    g = arr[:,:,1]
    b = arr[:,:,2]

    green_pixels = ((g > r) & (g > b)).mean()
    return green_pixels > 0.20

def predict_image(path):
    img = Image.open(path).convert("RGB")

    if not is_leaf(img):
        return "Invalid Image! Upload leaf image only.", 0

    img = img.resize((224,224))
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    pred = model.predict(img)[0]
    idx = np.argmax(pred)
    confidence = round(float(np.max(pred))*100,2)

    result = classes[idx]

    if "healthy" in result.lower():
        result = result + " (Healthy Leaf)"
    else:
        result = result + " (Disease Detected)"

    return result, confidence