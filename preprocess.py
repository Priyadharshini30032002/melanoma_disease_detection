import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, ImageDataGenerator
from PIL import Image, ImageTk

preprocessed_img = None

# Data augmentation settings
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

def preprocess_image(pil_img, status_label, display_label):
    global preprocessed_img
    if pil_img is None:
        status_label.config(text="❌ No image loaded!")
        return None

    # Step 1: Resize to 224x224
    img_resized = pil_img.resize((224, 224))

    # Step 2: Convert to array & normalize
    img_array = img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # shape (1, 224, 224, 3)

    # Step 3: Apply augmentation (take 1 augmented sample)
    aug_iter = datagen.flow(img_array, batch_size=1)
    img_aug = next(aug_iter)[0]  # (224, 224, 3)

    preprocessed_img = np.expand_dims(img_aug, axis=0)

    # Step 4: Show the preprocessed image in UI
    img_to_show = Image.fromarray((img_aug * 255).astype(np.uint8))
    tk_img = ImageTk.PhotoImage(img_to_show)
    display_label.config(image=tk_img, text="")   # clear placeholder text
    display_label.image = tk_img  # keep reference

    # Status text
    status_label.config(text=f"✅ Preprocessing done! Shape: {preprocessed_img.shape}")
    return preprocessed_img
