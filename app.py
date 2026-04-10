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

# ================= CONFIDENCE =================
CONFIDENCE_THRESHOLD = 85   # 🔥 Strong filter

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

        # FILE CHECK
        if 'image' not in request.files:
            return jsonify({"error": "No image key found"}), 400

        file = request.files.get('image')

        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        filepath = os.path.join(UPLOAD_DIR, file.filename)
        file.save(filepath)

        # ================= PREPROCESS =================
        img = load_img(filepath, target_size=(224, 224))
        arr = img_to_array(img) / 255.0
        arr = np.expand_dims(arr, axis=0)

        # ================= PREDICT =================
        preds = model.predict(arr, verbose=0)
        confidence = float(np.max(preds)) * 100
        top_idx = int(np.argmax(preds))
        top_class = classes.get(top_idx, "unknown")

        print(f"🔍 Prediction: {top_class}, Confidence: {confidence}")

        # ================= 🔥 NON-BEE FILTER =================
        if confidence < CONFIDENCE_THRESHOLD or top_class == "unknown":
            print("⚠️ Rejected (Not bee / low confidence)")
            return jsonify({
                "prediction": "invalid",
                "status": "Not a bee image",
                "severity": "none",
                "confidence": round(confidence, 2),
                "description": "Uploaded image is not a bee or unclear.",
                "action": "Please upload a clear bee image."
            })

        # ================= 🔥 SMART OTHER ISSUE =================
        if top_class == "other_issue":
            return jsonify({
                "prediction": "other_issue",
                "status": "Possible Issue Detected",
                "severity": "moderate",
                "confidence": round(confidence, 2),
                "description": "The system detected abnormalities but cannot confirm exact disease.",
                "possible_causes": [
                    "Ant attack",
                    "Colony stress",
                    "Environmental disturbance",
                    "Queen issues"
                ],
                "action": "Manual inspection recommended for accurate diagnosis."
            })

        # ================= NORMAL OUTPUT =================
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
            "confidence": round(confidence, 2),
            "description": info["description"],
            "action": info["action"]
        })

    except Exception as e:
        print("❌ ERROR:", str(e))
        return jsonify({"error": str(e)}), 500


# ================= RUN =================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
