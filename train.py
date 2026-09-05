
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models, callbacks
import json, os
import warnings

warnings.filterwarnings('ignore')

IMG_SIZE = (224,224)
BATCH_SIZE = 32

print("Loading dataset...")
train_gen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=25,
    zoom_range=0.2,
    horizontal_flip=True
)

train_data = train_gen.flow_from_directory(
    'dataset/',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_data = train_gen.flow_from_directory(
    'dataset/',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Save class names
class_names = list(train_data.class_indices.keys())
os.makedirs("models", exist_ok=True)
with open("models/class_names.json", "w") as f:
    json.dump(class_names, f, indent=2)

print(f"Found {len(class_names)} classes: {', '.join(class_names)}")

# Model
print("Creating model...")
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print(f"Model created. Total parameters: {model.count_params():,}")

# Callbacks
print("Starting training...")
cb = [
    callbacks.EarlyStopping(patience=3, restore_best_weights=True, verbose=1),
    callbacks.ModelCheckpoint("models/plant_model.h5", save_best_only=True, verbose=1)
]

history = model.fit(train_data, validation_data=val_data, epochs=10, callbacks=cb, verbose=1)

# Save final
print("Saving model...")
model.save("models/plant_model.h5")

# Plot accuracy
print("Generating accuracy plot...")
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("models/accuracy.png", dpi=150, bbox_inches='tight')
print("Training complete! Model saved to models/plant_model.h5")
print(f"Accuracy plot saved to models/accuracy.png")
print(f"Class names saved to models/class_names.json")
