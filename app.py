from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import json
import os

app = Flask(__name__)

# Load model and class labels
MODEL_PATH = 'model/deepcrop_cnn.h5'
CLASS_INDICES_PATH = 'model/class_indices.json'

print("[DIAG] Loading model...")
model = load_model(MODEL_PATH)
print("[DIAG] Model loaded.")

with open(CLASS_INDICES_PATH, "r") as f:
    class_indices = json.load(f)

classes = list(class_indices.keys())
print(f"[DIAG] Loaded {len(classes)} classes.")

# Mapping of diseases to recommended products
product_links = {
    "Apple___Black_rot": "https://www.amazon.in/s?k=apple+fungicide",
    "Apple___healthy": "https://www.amazon.in/s?k=organic+fertilizer",
    "Corn___Common_rust": "https://www.amazon.in/s?k=corn+fungicide",
    "Corn___healthy": "https://www.amazon.in/s?k=corn+nutrients",
    "Grape___Black_rot": "https://www.amazon.in/s?k=grape+disease+control",
    "Grape___healthy": "https://www.amazon.in/s?k=grape+fertilizer",
    "Potato___Early_blight": "https://www.amazon.in/s?k=potato+fungicide",
    "Potato___Late_blight": "https://www.amazon.in/s?k=potato+disease+control",
    "Potato___healthy": "https://www.amazon.in/s?k=potato+fertilizer",
    "Tomato___Leaf_Mold": "https://www.amazon.in/s?k=tomato+fungicide",
    "Tomato___healthy": "https://www.amazon.in/s?k=tomato+nutrient",
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    img = request.files['image']
    if img.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save and preprocess image
    img_path = os.path.join('static', 'uploaded.jpg')
    img.save(img_path)
    img_load = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img_load) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    preds = model.predict(img_array)
    class_id = np.argmax(preds, axis=1)[0]
    result = classes[class_id]
    confidence = float(np.max(preds))

    product_url = product_links.get(result, "https://www.amazon.in/s?k=plant+care")

    return render_template(
        'index.html',
        prediction=result,
        confidence=round(confidence * 100, 2),
        image_path=img_path,
        product_link=product_url
    )

if __name__ == '__main__':
    app.run(debug=True)
