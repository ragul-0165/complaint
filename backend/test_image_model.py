import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

model = tf.keras.models.load_model("models/image_model.h5")

# 🔥 ADD THIS
class_labels = ['Damaged concrete', 'Graffiti', 'IllegalParking', 'Potholes', 'Garbage']

img_path = input("Enter image path:")

img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)

# 🔥 CONVERT TO LABEL
predicted_class = class_labels[np.argmax(prediction)]

print("Prediction:", predicted_class)