import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20

# Data Preprocessing with Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    r"C:\Users\91944\OneDrive\Desktop\Melanoma_disease_detection\dataset\melanoma\train",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    r"C:\Users\91944\OneDrive\Desktop\Melanoma_disease_detection\dataset\melanoma\test",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

# Custom CNN Architecture (based on paper)
model = Sequential([
    Conv2D(64, (3,3), strides=(2,2), activation='relu', input_shape=(224,224,3), padding='valid'),
    MaxPooling2D(pool_size=(2,2), padding='valid'),
    
    Conv2D(64, (3,3), strides=(2,2), activation='relu', padding='valid'),
    MaxPooling2D(pool_size=(2,2), padding='valid'),
    
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # binary classification
])

# Compile model
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS
)

# Save model
model.save("cnn_skin_model.h5")

print("Training complete. Model saved as cnn_skin_model.h5")
