import joblib
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

print("🚀 Loading models...")

# Load models
text_model = joblib.load("models/text_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")
image_model = tf.keras.models.load_model("models/image_model.h5")

# Image labels
class_labels = ['Damaged concrete', 'Graffiti', 'IllegalParking', 'Potholes', 'Garbage']

print("✅ Models loaded")

# -------------------------------
# TEXT PREDICTION
# -------------------------------
def predict_text(text):
    vec = vectorizer.transform([text])
    pred = text_model.predict(vec)
    return pred[0]

# -------------------------------
# IMAGE PREDICTION
# -------------------------------
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = image_model.predict(img_array)
    return class_labels[np.argmax(prediction)]

# -------------------------------
# MAIN SYSTEM
# -------------------------------
if __name__ == "__main__":

    print("\n🔹 SELECT INPUT TYPE")
    print("1. Text Complaint (Banking)")
    print("2. Image Complaint (Civic)")

    choice = input("Enter choice (1/2): ")

    if choice == "1":
        text = input("\nEnter complaint: ")
        result = predict_text(text)
        print("\n✅ Predicted Issue:", result)

    elif choice == "2":
        path = input("\nEnter image path: ")
        result = predict_image(path)
        print("\n✅ Predicted Issue:", result)

    else:
        print("❌ Invalid choice")