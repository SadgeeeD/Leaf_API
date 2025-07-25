from flask import Flask, request, jsonify
from keras.models import load_model
import numpy as np
from PIL import Image
import base64
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

app = Flask(__name__)

# Load both models at startup
species_model = load_model("leaf_model2.keras")
health_model = load_model("leaf_model2.keras")

# Class labels
species_classes = ["bok choy", "nai bai", "peppermint"]
health_classes = [
    "anthracnose", "downy_mildew", "fusarium_leaf_spot", "healthy",
    "leaf_spot", "powdery_mildew", "viral_mosaic"
]

# Image preprocessing with compression
def preprocess_image(base64_image):
    try:
        # Decode and open image
        img = Image.open(io.BytesIO(base64.b64decode(base64_image)))
        img = img.resize((180, 180)).convert("RGB")

        # Compress the image to reduce file size
        compressed_io = io.BytesIO()
        img.save(compressed_io, format="JPEG", quality=70, optimize=True)
        compressed_io.seek(0)
        compressed_img = Image.open(compressed_io)

        # Normalize
        img_array = np.array(compressed_img) / 255.0
        return np.expand_dims(img_array, axis=0)

    except Exception as e:
        print("❌ Error during image processing:", e)
        raise

@app.route("/")
def home():
    return jsonify({"message": "✅ Multi-model prediction server is online."})

@app.route("/predict/species", methods=["POST"])
def predict_species():
    try:
        data = request.json
        print("📥 Raw incoming data:", data)

        if not data or "image" not in data:
            raise ValueError("Missing 'image' in request JSON")

        img_array = preprocess_image(data["image"])
        prediction = species_model.predict(img_array)[0]
        index = int(np.argmax(prediction))

        return jsonify({
            "predicted_class": species_classes[index],
            "confidence": float(round(prediction[index], 4))
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"{type(e).__name__}: {str(e)}"}), 500

@app.route("/predict/health", methods=["POST"])
def predict_health():
    try:
        data = request.json
        img_array = preprocess_image(data["image"])
        prediction = health_model.predict(img_array)[0]
        index = int(np.argmax(prediction))

        return jsonify({
            "predicted_class": health_classes[index],
            "confidence": float(round(prediction[index], 4))
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6000)
