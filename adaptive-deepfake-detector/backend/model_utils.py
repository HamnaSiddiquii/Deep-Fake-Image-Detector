import tensorflow as tf
import numpy as np
from PIL import Image
import os

# The directory where model_utils.py is located (the backend folder)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Point directly to where model.keras lives
MODEL_PATH = os.path.join(BASE_DIR, "model.keras")

# If your model.keras file is out in the MAIN root folder instead, use this:
# MODEL_PATH = os.path.join(os.path.dirname(BASE_DIR), "model.keras")

IMG_SIZE = 224

model = None

def load_model_once():
    global model
    if model is None:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Model loaded successfully")


    # Your remaining image preprocessing and model.predict(img) logic goes here...
def predict_image(image_path):


    load_model_once()

    img = Image.open(image_path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))

    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img, verbose=0)[0][0]

    if pred > 0.5:
        result = "Real Image"
    else:
        result = "AI Generated (Fake)"

    return result, float(pred)
