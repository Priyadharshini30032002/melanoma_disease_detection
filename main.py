import tkinter as tk
from tkinter import filedialog, scrolledtext
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, ImageDataGenerator
from feature_extraction import extract_features  # Import the feature extraction function

# Global variables
loaded_image = None
preprocessed_img = None
extracted_features = None

# Data augmentation
datagen = ImageDataGenerator(
    # rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

# Preprocessing function
def preprocess_image(pil_img, display_label, status_label):
    global preprocessed_img
    if pil_img is None:
        status_label.config(text="❌ No image loaded!")
        return None

    # Resize to 224x224 for model input
    img_resized = pil_img.resize((224, 224))

    # Convert to array & normalize
    img_array = img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # (1,224,224,3)

    # Apply augmentation (take 1 sample)
    aug_iter = datagen.flow(img_array, batch_size=1)
    img_aug = next(aug_iter)[0]  # (224,224,3)

    preprocessed_img = np.expand_dims(img_aug, axis=0)

    # Show the preprocessed image in Tkinter
    # Convert back to PIL Image and resize for better display
    display_img = Image.fromarray((img_aug * 255).astype(np.uint8))
    
    # Resize for display while maintaining aspect ratio
    max_display_size = (400, 400)
    display_img.thumbnail(max_display_size, Image.Resampling.LANCZOS)
    
    tk_img = ImageTk.PhotoImage(display_img)
    display_label.config(image=tk_img)
    display_label.image = tk_img

    status_label.config(text=f"✅ Preprocessing done! Shape: {preprocessed_img.shape}")

# Feature extraction function
def perform_feature_extraction(features_text, status_label):
    global preprocessed_img, extracted_features
    
    if preprocessed_img is None:
        status_label.config(text="❌ Please preprocess an image first!")
        return
    
    try:
        # Extract features
        features, features_shape = extract_features(preprocessed_img)
        extracted_features = features
        
        # Display feature information
        features_text.config(state=tk.NORMAL)
        features_text.delete(1.0, tk.END)
        features_text.insert(tk.END, f"Feature Extraction Results:\n")
        features_text.insert(tk.END, f"Feature shape: {features_shape}\n")
        features_text.insert(tk.END, f"Flattened features length: {len(features)}\n")
        features_text.insert(tk.END, f"First 10 feature values:\n")
        features_text.insert(tk.END, f"{features[:10]}\n")
        features_text.config(state=tk.DISABLED)
        
        status_label.config(text=f"✅ Feature extraction complete! Extracted {len(features)} features.")
        
    except Exception as e:
        status_label.config(text=f"❌ Error during feature extraction: {str(e)}")

# App class
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Melanoma Detection")

        # Create frames for better organization
        self.original_frame = tk.LabelFrame(root, text="Original Image")
        self.original_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        self.processed_frame = tk.LabelFrame(root, text="Preprocessed Image")
        self.processed_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        
        # Configure grid weights for responsive resizing
        root.grid_columnconfigure(0, weight=1)
        root.grid_columnconfigure(1, weight=1)
        root.grid_rowconfigure(0, weight=1)
        root.grid_rowconfigure(1, weight=1)

        # Original image label
        self.img_label = tk.Label(self.original_frame, text="No image loaded", bg='lightgray')
        self.img_label.pack(padx=10, pady=10, expand=True, fill=tk.BOTH)

        # Preprocessed image label
        self.pre_img_label = tk.Label(self.processed_frame, text="Preprocessed image will appear here", bg='lightgray')
        self.pre_img_label.pack(padx=10, pady=10, expand=True, fill=tk.BOTH)

        # Features frame
        self.features_frame = tk.LabelFrame(root, text="Feature Extraction Results")
        self.features_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")
        
        # Features text widget
        self.features_text = scrolledtext.ScrolledText(self.features_frame, height=8, state=tk.DISABLED)
        self.features_text.pack(padx=10, pady=10, expand=True, fill=tk.BOTH)
        self.features_text.config(state=tk.NORMAL)
        self.features_text.insert(tk.END, "Feature extraction results will appear here after processing.")
        self.features_text.config(state=tk.DISABLED)

        # Status label
        self.status_label = tk.Label(root, text="", fg="green")
        self.status_label.grid(row=2, column=0, columnspan=2, pady=5)

        # Buttons frame
        self.button_frame = tk.Frame(root)
        self.button_frame.grid(row=3, column=0, columnspan=2, pady=10)

        # Buttons
        self.load_btn = tk.Button(self.button_frame, text="Load Image", command=self.load_image, width=15)
        self.load_btn.pack(side=tk.LEFT, padx=5)

        self.preprocess_btn = tk.Button(self.button_frame, text="Preprocess", command=self.preprocess_image, width=15)
        self.preprocess_btn.pack(side=tk.LEFT, padx=5)

        self.feature_btn = tk.Button(self.button_frame, text="Extract Features", command=self.extract_features, width=15)
        self.feature_btn.pack(side=tk.LEFT, padx=5)

    def load_image(self):
        global loaded_image
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        if file_path:
            loaded_image = Image.open(file_path)
            
            # Create a copy for display with appropriate size
            display_img = loaded_image.copy()
            
            # Resize for display while maintaining aspect ratio
            max_display_size = (400, 400)
            display_img.thumbnail(max_display_size, Image.Resampling.LANCZOS)
            
            tk_img = ImageTk.PhotoImage(display_img)
            self.img_label.config(image=tk_img, text="")
            self.img_label.image = tk_img
            self.status_label.config(text="🖼 Original image loaded.")

    def preprocess_image(self):
        if loaded_image is None:
            self.status_label.config(text="❌ Please load an image first")
        else:
            preprocess_image(loaded_image, self.pre_img_label, self.status_label)

    def extract_features(self):
        perform_feature_extraction(self.features_text, self.status_label)

# Run the app
if __name__ == "__main__":
    root = tk.Tk()
    # Set a minimum window size
    root.minsize(1000, 900)  # Increased size to accommodate feature display
    app = App(root)
    root.mainloop()

# import tkinter as tk
# from tkinter import filedialog
# from PIL import Image, ImageTk
# import numpy as np
# from tensorflow.keras.preprocessing.image import img_to_array, ImageDataGenerator

# # Global variables
# loaded_image = None
# preprocessed_img = None

# # Data augmentation
# datagen = ImageDataGenerator(
#     rotation_range=20,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     shear_range=0.1,
#     zoom_range=0.1,
#     horizontal_flip=True
# )

# # Preprocessing function
# def preprocess_image(pil_img, display_label, status_label):
#     global preprocessed_img
#     if pil_img is None:
#         status_label.config(text="❌ No image loaded!")
#         return None

#     # Resize to 224x224 for model input
#     img_resized = pil_img.resize((224, 224))

#     # Convert to array & normalize
#     img_array = img_to_array(img_resized) / 255.0
#     img_array = np.expand_dims(img_array, axis=0)  # (1,224,224,3)

#     # Apply augmentation (take 1 sample)
#     aug_iter = datagen.flow(img_array, batch_size=1)
#     img_aug = next(aug_iter)[0]  # (224,224,3)

#     preprocessed_img = np.expand_dims(img_aug, axis=0)

#     # Show the preprocessed image in Tkinter
#     # Convert back to PIL Image and resize for better display
#     display_img = Image.fromarray((img_aug * 255).astype(np.uint8))
    
#     # Resize for display while maintaining aspect ratio
#     max_display_size = (400, 400)  # Increased size for better visibility
#     display_img.thumbnail(max_display_size, Image.Resampling.LANCZOS)
    
#     tk_img = ImageTk.PhotoImage(display_img)
#     display_label.config(image=tk_img)
#     display_label.image = tk_img

#     status_label.config(text=f"✅ Preprocessing done! Shape: {preprocessed_img.shape}")

# # App class
# class App:
#     def __init__(self, root):
#         self.root = root
#         self.root.title("Melanoma Detection")

#         # Create frames for better organization
#         self.original_frame = tk.LabelFrame(root, text="Original Image")
#         self.original_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
#         self.processed_frame = tk.LabelFrame(root, text="Preprocessed Image")
#         self.processed_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        
#         # Configure grid weights for responsive resizing
#         root.grid_columnconfigure(0, weight=1)
#         root.grid_columnconfigure(1, weight=1)
#         root.grid_rowconfigure(0, weight=1)

#         # Original image label - REMOVED width and height parameters
#         self.img_label = tk.Label(self.original_frame, text="No image loaded", bg='lightgray')
#         self.img_label.pack(padx=10, pady=10, expand=True, fill=tk.BOTH)

#         # Preprocessed image label - REMOVED width and height parameters
#         self.pre_img_label = tk.Label(self.processed_frame, text="Preprocessed image will appear here", bg='lightgray')
#         self.pre_img_label.pack(padx=10, pady=10, expand=True, fill=tk.BOTH)

#         # Status label
#         self.status_label = tk.Label(root, text="", fg="green")
#         self.status_label.grid(row=1, column=0, columnspan=2, pady=10)

#         # Buttons frame
#         self.button_frame = tk.Frame(root)
#         self.button_frame.grid(row=2, column=0, columnspan=2, pady=10)

#         # Buttons
#         self.load_btn = tk.Button(self.button_frame, text="Load Image", command=self.load_image, width=20)
#         self.load_btn.pack(side=tk.LEFT, padx=10)

#         self.preprocess_btn = tk.Button(self.button_frame, text="Preprocess", command=self.preprocess_image, width=20)
#         self.preprocess_btn.pack(side=tk.LEFT, padx=10)

#     def load_image(self):
#         global loaded_image
#         file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
#         if file_path:
#             loaded_image = Image.open(file_path)
            
#             # Create a copy for display with appropriate size
#             display_img = loaded_image.copy()
            
#             # Resize for display while maintaining aspect ratio
#             max_display_size = (400, 400)  # Increased size for better visibility
#             display_img.thumbnail(max_display_size, Image.Resampling.LANCZOS)
            
#             tk_img = ImageTk.PhotoImage(display_img)
#             self.img_label.config(image=tk_img, text="")
#             self.img_label.image = tk_img
#             self.status_label.config(text="🖼 Original image loaded.")

#     def preprocess_image(self):
#         if loaded_image is None:
#             self.status_label.config(text="❌ Please load an image first")
#         else:
#             preprocess_image(loaded_image, self.pre_img_label, self.status_label)

# # Run the app
# if __name__ == "__main__":
#     root = tk.Tk()
#     # Set a minimum window size
#     root.minsize(900, 700)  # Increased minimum size for better image display
#     app = App(root)
#     root.mainloop()