from flask import Flask, request, jsonify
from keras.models import load_model
import numpy as np
from PIL import Image
import base64
import io

app = Flask(__name__)

# Load both models at startup
species_model = load_model("species_model.keras")
health_model = load_model("leaf_model.keras")

# Define your class labels here (in the same order as your training)
species_classes = ["bok choy", "nai bai", "peppermint"]
health_classes = [
    "anthracnose", "downy_mildew", "fusarium_leaf_spot", "healthy",
    "leaf_spot", "powdery_mildew", "viral_mosaic"
]



# Image preprocessing
def preprocess_image(base64_image):
    img = Image.open(io.BytesIO(base64.b64decode(base64_image)))
    img = img.resize((180, 180)).convert("RGB")
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

@app.route("/")
def home():
    return jsonify({"message": "✅ Multi-model prediction server is online."})

@app.route("/predict/species", methods=["POST"])
def predict_species():
    try:
        data = request.json
        img_array = preprocess_image(data["image"])
        prediction = species_model.predict(img_array)[0]
        index = int(np.argmax(prediction))
        return jsonify({
            "class": species_classes[index],
            "confidence": float(round(prediction[index], 4))
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict/health", methods=["POST"])
def predict_health():
    try:
        data = request.json
        img_array = preprocess_image(data["image"])
        prediction = health_model.predict(img_array)[0]
        index = int(np.argmax(prediction))
        return jsonify({
            "class": health_classes[index],
            "confidence": float(round(prediction[index], 4))
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)