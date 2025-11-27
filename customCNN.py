import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Enable Mixed Precision for better GPU performance
tf.keras.mixed_precision.set_global_policy("mixed_float16")

# Ensure TensorFlow Uses GPU Efficiently
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPUs successfully configured.")
    except RuntimeError as e:
        print(e)

# Define Dataset Path
dataset_path = "C:\\Users\\kesha\\OneDrive\\Desktop\\brain dataset\\converted_images"

# Define image size and batch size
img_size = (128, 128)
batch_size = 64

# Data Augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    validation_split=0.2
)

# Load Training and Validation Data
train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='grayscale',
    subset='training'
)

val_data = datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='grayscale',
    subset='validation',
    shuffle=False  # Important for correct label extraction
)

# Get Class Names
class_names = list(train_data.class_indices.keys())
print("Class Names:", class_names)

# Define CNN Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 1)),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(256, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),

    Dense(256, activation='relu'),
    Dropout(0.3),

    Dense(len(class_names), activation='softmax', dtype='float32')  # Output layer (3 classes)
])

# Enable XLA for faster training
tf.config.optimizer.set_jit(True)

# Compile Model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train Model
history = model.fit(train_data, validation_data=val_data, epochs=30)

# Evaluate Model
val_loss, val_acc = model.evaluate(val_data)
print(f"Validation Accuracy: {val_acc:.2%}")

# Get True Labels
true_labels = val_data.classes

# Get Predicted Labels
predictions = model.predict(val_data)
predicted_labels = np.argmax(predictions, axis=1)

# Print Classification Report (Precision, Recall, F1-Score)
report = classification_report(true_labels, predicted_labels, target_names=class_names)
print("\nClassification Report:\n", report)
