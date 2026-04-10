from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import json, os
import tensorflow as tf
# Temporarily disable model loading
model = None
app = Flask(__name__)
CORS(app)

BASE_PATH  = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_PATH, "bee_model.keras")
CLASS_PATH = os.path.join(BASE_PATH, "class_names.json")
UPLOAD_DIR = os.path.join(BASE_PATH, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
with open(CLASS_PATH) as f:
    class_indices = json.load(f)
classes = {v: k for k, v in class_indices.items()}
print("✅ Model ready:", classes)

disease_info = {
    "healthy": {
        "status"     : "Healthy Colony",
        "severity"   : "low",
        "description": "Your bee colony appears healthy. No disease detected.",
        "action"     : "Continue regular monitoring every 2 weeks."
    },
    "varroa": {
        "status"     : "Varroa Mite Infestation",
        "severity"   : "high",
        "description": "Varroa mites detected. These parasites weaken bees and spread deadly viruses.",
        "action"     : "Apply oxalic acid treatment immediately. Re-inspect after 7 days."
    },
    "other_issue": {
        "status"     : "Colony Stress Detected",
        "severity"   : "moderate",
        "description": "Signs of colony stress — possible ant problem, robbing, or missing queen.",
        "action"     : "Physically inspect the hive. Check queen presence and entrance security."
    }
}

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image"}), 400
    file     = request.files['image']
    filepath = os.path.join(UPLOAD_DIR, file.filename)
    file.save(filepath)
    img  = keras_image.load_img(filepath, target_size=(224, 224))
    arr  = keras_image.img_to_array(img) / 255.0
    arr  = np.expand_dims(arr, axis=0)
    preds     = model.predict(arr, verbose=0)
    top_idx   = int(np.argmax(preds))
    top_class = classes[top_idx]
    confidence= round(float(np.max(preds)) * 100, 2)
    info      = disease_info[top_class]
    return jsonify({
        "prediction" : top_class,
        "status"     : info["status"],
        "severity"   : info["severity"],
        "confidence" : confidence,
        "description": info["description"],
        "action"     : info["action"],
        "all_scores" : {
            classes[i]: round(float(preds[0][i]) * 100, 2)
            for i in range(len(classes))
        }
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "running"})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)