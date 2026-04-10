from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import json
import os
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array

app = Flask(__name__)
CORS(app)

# ================= PATH SETUP =================
BASE_PATH  = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_PATH, "bee_model.h5")
CLASS_PATH = os.path.join(BASE_PATH, "class_names.json")
UPLOAD_DIR = os.path.join(BASE_PATH, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ================= LOAD CLASSES =================
try:
    with open(CLASS_PATH) as f:
        class_indices = json.load(f)
    classes = {v: k for k, v in class_indices.items()}
    print("📊 Classes loaded:", classes)
except Exception as e:
    print("❌ Class file error:", e)
    classes = {}

# ================= LAZY MODEL LOAD =================
model = None

def load_model_once():
    global model
    if model is None:
        print("🚀 Loading model...")
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Model loaded")

# ================= DISEASE INFO =================
disease_info = {
    "healthy": {
        "status": "Healthy Colony",
        "severity": "low",
        "description": "Your bee colony appears healthy.",
        "action": "Continue monitoring."
    },
    "varroa": {
        "status": "Varroa Mite Infestation",
        "severity": "high",
        "description": "Dangerous mites detected.",
        "action": "Apply treatment immediately."
    },
    "other_issue": {
        "status": "Colony Stress",
        "severity": "medium",
        "description": "Possible colony stress.",
        "action": "Inspect hive."
    }
}

# ================= ROOT =================
@app.route('/')
def home():
    return "🚀 Kanasu Bee House API is LIVE!"

# ================= HEALTH =================
@app.route('/health')
def health():
    return jsonify({
        "status": "running",
        "model_loaded": model is not None
    })

# ================= PREDICT =================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        load_model_once()

        if model is None:
            return jsonify({"error": "Model not loaded"}), 500

        # 🔥 FIXED FILE HANDLING
        if 'image' not in request.files:
            return jsonify({"error": "No image key found"}), 400

        file = request.files.get('image')

        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        filepath = os.path.join(UPLOAD_DIR, file.filename)
        file.save(filepath)

        # Preprocess
        img = load_img(filepath, target_size=(224, 224))
        arr = img_to_array(img) / 255.0
        arr = np.expand_dims(arr, axis=0)

        # Predict
        preds = model.predict(arr, verbose=0)
        top_idx = int(np.argmax(preds))
        top_class = classes.get(top_idx, "unknown")
        confidence = round(float(np.max(preds)) * 100, 2)

        info = disease_info.get(top_class, {
            "status": "Unknown",
            "severity": "unknown",
            "description": "No info available",
            "action": "Check manually"
        })

        return jsonify({
            "prediction": top_class,
            "status": info["status"],
            "severity": info["severity"],
            "confidence": confidence,
            "description": info["description"],
            "action": info["action"]
        })

    except Exception as e:
        print("❌ ERROR:", str(e))  # VERY IMPORTANT
        return jsonify({"error": str(e)}), 500


# ================= RUN =================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
