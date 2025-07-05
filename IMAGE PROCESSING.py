#IMAGE PROCESSING
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define paths
train_dir = '/content/drive/MyDrive/College-Tag/train'
test_dir = '/content/drive/MyDrive/College-Tag/test'

# Image size and batch size
IMG_SIZE = 128
BATCH_SIZE = 32

# Data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

# Prepare train and validation data
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)
from google.colab import drive
drive.mount('/content/drive')


#MODEL CREATION(CNN WITH ATTENTION MECHANISM)
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, Multiply, GlobalAveragePooling2D
from tensorflow.keras.models import Model

def attention_block(inputs):
    attention = Conv2D(64, (1, 1), activation='relu')(inputs)
    attention = GlobalAveragePooling2D()(attention)
    attention = Dense(128, activation='sigmoid')(attention)
    attention = Multiply()([inputs, attention])
    return attention
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define paths for training and validation datasets
train_dir = '/content/drive/MyDrive/College-Tag/train'
validation_dir = '/content/drive/MyDrive/College-Tag/test'

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

validation_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=32,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=32,
    class_mode='categorical'
)

# Now train the model
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator
)


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout


IMG_SIZE = 128
input_img = Input(shape=(IMG_SIZE, IMG_SIZE, 3))

# Convolutional layers
x = Conv2D(32, (3, 3), activation='relu')(input_img)
x = MaxPooling2D(pool_size=(2, 2))(x)

# Further Convolutional layers
x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

# Flatten and Dense layers
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)

# Output layer (for 8 classes)
color_output = Dense(8, activation='softmax', name='color_output')(x)

# Define model
model = Model(inputs=input_img, outputs=color_output)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Now you can train the model
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator
)

#MODEL PREDICTION
#PREPROCESSING IMAGE
import numpy as np
from tensorflow.keras.preprocessing import image

def preprocess_input_image(img_path):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize pixel values
    return img_array
#PREDICTION
# Load and preprocess new image
img_path = '/content/drive/MyDrive/College-Tag/test/it/it-14.jpg'
new_img = preprocess_input_image(img_path)

# Get model predictions
predictions = model.predict(new_img)
predicted_class = np.argmax(predictions, axis=1)

# Map class index to color label
class_labels = list(train_generator.class_indices.keys())
predicted_color = class_labels[predicted_class[0]]
print(f"Predicted Branch: {predicted_color}")


#PLOTTING TRAINING AND VALIDATION PERFORMANCE
import matplotlib.pyplot as plt

# Plot accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#SAVE TRAIN MODEL
# Save the model
model.save('color_tag_classifier_with_attention.h5')
