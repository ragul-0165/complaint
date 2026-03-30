from flask import Flask, request, render_template
import joblib
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)

# -----------------------------
# LOAD MODELS
# -----------------------------
text_model = joblib.load("models/text_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")
image_model = tf.keras.models.load_model("models/image_model.h5")

# Image classes
class_labels = ['Damaged concrete', 'Graffiti', 'IllegalParking', 'Potholes', 'Garbage']

# -----------------------------
# 🔥 DEPARTMENT SIMULATION
# -----------------------------
def get_department_info(label):

    # Banking related
    if "Account" in label or "Transaction" in label:
        return "🏦 Banking Department", "Upload bank statement / transaction proof", "3-5 days"

    # Civic related
    elif label == "Potholes":
        return "🚧 Road & Traffic Department", "Provide location details", "2-3 days"

    elif label == "Garbage":
        return "🗑 Municipal Department", "Upload area photo / location", "1-2 days"

    elif label == "IllegalParking":
        return "🚓 Traffic Police Department", "Provide vehicle details / location", "2-3 days"

    elif label == "Graffiti":
        return "🏙 Municipal Cleaning Department", "Upload clear image of area", "2-4 days"

    elif label == "Damaged concrete":
        return "🏗 Infrastructure Department", "Provide location and damage details", "4-6 days"

    else:
        return "📌 General Department", "Provide more details", "3 days"


# -----------------------------
# 🔥 PRIORITY RULE
# -----------------------------
def get_priority(text):
    text = text.lower()
    if "urgent" in text or "fraud" in text:
        return "High"
    elif "delay" in text:
        return "Medium"
    else:
        return "Low"


# -----------------------------
# HOME PAGE
# -----------------------------
@app.route('/')
def home():
    return render_template('index.html')


# -----------------------------
# TEXT PREDICTION
# -----------------------------
@app.route('/predict_text', methods=['POST'])
def predict_text():
    text = request.form['text']

    # 🔥 Predict
    vec = vectorizer.transform([text])
    prediction = text_model.predict(vec)[0]

    # 🔥 Get department info
    dept, suggestion, time = get_department_info(prediction)

    # 📂 Handle optional supporting documents
    docs_msg = ""
    if 'docs' in request.files:
        doc_file = request.files['docs']
        if doc_file.filename != '':
            doc_path = os.path.join("uploads", "doc_" + doc_file.filename)
            os.makedirs("uploads", exist_ok=True)
            doc_file.save(doc_path)
            docs_msg = "<br><small>✅ <i>Supporting document attached</i></small>"

    # 🔥 Get Priority
    priority = get_priority(text)

    result = f"""
    <b>📝 Text Prediction:</b> {prediction} <br><br>
    <b>🏢 Department:</b> {dept} <br>
    <b>💡 Suggestion:</b> {suggestion} <br>
    <b>⏳ Estimated Time:</b> {time} <br>
    <b>🔥 Priority:</b> {priority}
    {docs_msg}
    """

    return render_template('index.html', result=result)


# -----------------------------
# IMAGE PREDICTION
# -----------------------------
@app.route('/predict_image', methods=['POST'])
def predict_image():
    file = request.files['image']

    filepath = os.path.join("uploads", file.filename)
    os.makedirs("uploads", exist_ok=True)
    file.save(filepath)

    # 🔥 Predict
    img = image.load_img(filepath, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = image_model.predict(img_array)
    label = class_labels[np.argmax(prediction)]

    # 📂 Handle optional supporting documents
    docs_msg = ""
    if 'docs' in request.files:
        doc_file = request.files['docs']
        if doc_file.filename != '':
            doc_path = os.path.join("uploads", "doc_img_" + doc_file.filename)
            os.makedirs("uploads", exist_ok=True)
            doc_file.save(doc_path)
            docs_msg = "<br><small>✅ <i>Supporting document attached</i></small>"

    # 🔥 Get department info
    dept, suggestion, time = get_department_info(label)

    # 🔥 Get Priority (using label as text)
    priority = get_priority(label)

    result = f"""
    <b>📷 Image Prediction:</b> {label} <br><br>
    <b>🏢 Department:</b> {dept} <br>
    <b>💡 Suggestion:</b> {suggestion} <br>
    <b>⏳ Estimated Time:</b> {time} <br>
    <b>🔥 Priority:</b> {priority}
    {docs_msg}
    """

    return render_template('index.html', result=result)


# -----------------------------
# RUN APP
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)