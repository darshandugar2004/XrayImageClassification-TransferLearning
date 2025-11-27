import os
import numpy as np
import tensorflow as tf
import scipy.io
import h5py
import cv2
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Flatten, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score

# Enable mixed precision to reduce memory usage
from tensorflow.keras.mixed_precision import experimental as mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

# Ensure TensorFlow uses GPU if available
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'  # Prevent fragmentation
        print("✅ GPU is being used for training with optimized memory management.")
    except RuntimeError as e:
        print(e)

# Path to your dataset folder
dataset_path = "C:\\Users\\kesha\\OneDrive\\Desktop\\brain dataset\\dataset\\data"

# Function to load MATLAB .mat files
def load_mat_data(path):
    images = []
    labels = []
    
    for file in os.listdir(path):
        if not file.endswith(".mat"):
            continue  # Skip non-mat files
        
        file_path = os.path.join(path, file)
        try:
            mat = scipy.io.loadmat(file_path)
            if 'cjdata' not in mat:
                print(f"Skipping {file}: 'cjdata' key not found!")
                continue
            
            image = mat['cjdata']['image'][0, 0]  # Extract the image
            label = int(mat['cjdata']['label'][0, 0]) - 1  # Convert label (1,2,3) -> (0,1,2)
        except NotImplementedError:
            with h5py.File(file_path, 'r') as mat:
                if 'cjdata' not in mat:
                    print(f"Skipping {file}: 'cjdata' key not found!")
                    continue
                image = np.array(mat['cjdata']['image'])
                label = int(np.array(mat['cjdata']['label']).item()) - 1
        
        # Resize and normalize image
        image = cv2.resize(image, (256, 256))
        image = image / 255.0  # Normalize
        
        # Convert grayscale to RGB (needed for InceptionResNetV2)
        image = np.stack((image,) * 3, axis=-1)

        images.append(image)
        labels.append(label)
        print(f"✅ Successfully extracted image and label from {file}")
    
    return np.array(images), np.array(labels)

# Load dataset
X, y = load_mat_data(dataset_path)

# Check if data is loaded correctly
if len(X) == 0 or len(y) == 0:
    raise ValueError("Error: No valid .mat files found or extraction failed!")

# Split dataset into train and validation sets (80% train, 20% val)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Data Augmentation
data_gen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

train_data = data_gen.flow(X_train, y_train, batch_size=8)  # Reduce batch size
val_data = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(8)

# Define input layer
input_layer = Input(shape=(256, 256, 3))

# Load InceptionResNetV2 without top layers
base_model = InceptionResNetV2(weights='imagenet', include_top=False, input_tensor=input_layer)
for layer in base_model.layers[-50:]:
    layer.trainable = True  # Unfreeze last 50 layers for fine-tuning

# Custom Dense Layers (DDIRNet as described in the research paper)
x = GlobalAveragePooling2D()(base_model.output)
x = Flatten()(x)
x = Dense(1024, activation='relu')(x)  # Reduce layer size
x = Dropout(0.4)(x)  # Increase dropout to prevent overfitting
out = Dense(3, activation='softmax', dtype='float32')(x)  # Ensure output is FP32

# Define the model
model = Model(inputs=input_layer, outputs=out)

# Learning rate scheduler
def lr_schedule(epoch):
    return 0.0001 * (0.1 ** (epoch // 10))
lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule)

# Compile model
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=20,  # Increased to 20 epochs
    verbose=1,
    callbacks=[lr_callback]
)

# Save model
model.save("brain_tumor_model.h5")

# Evaluate model
y_true = []
y_pred = []

for images, labels in val_data:
    preds = model.predict(images)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

print(f'📊 Accuracy: {accuracy * 100:.2f}%')
print(f'🎯 Precision: {precision:.4f}')
print(f'🔥 F1 Score: {f1:.4f}')
