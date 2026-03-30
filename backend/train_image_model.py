import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

print("🚀 Starting Image Model Training...")

# Dataset path
data_dir = "dataset/civic_classification"

# Image settings
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# -------------------------------
# 🔥 DATA GENERATOR
# -------------------------------

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = train_datagen.flow_from_directory(
    data_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_data = train_datagen.flow_from_directory(
    data_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

print("Classes:", train_data.class_indices)

# -------------------------------
# 🔥 LOAD PRETRAINED MODEL
# -------------------------------

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze base layers
base_model.trainable = False

# -------------------------------
# 🔥 CUSTOM LAYERS
# -------------------------------

x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.5)(x)
output = layers.Dense(train_data.num_classes, activation='softmax')(x)

model = models.Model(inputs=base_model.input, outputs=output)

# -------------------------------
# 🔥 COMPILE
# -------------------------------

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# -------------------------------
# 🔥 TRAIN MODEL
# -------------------------------

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=5
)

# -------------------------------
# 🔥 SAVE MODEL
# -------------------------------

model.save("models/image_model.h5")

print("💾 Image model saved successfully!")