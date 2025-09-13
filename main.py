# main.py
import tkinter as tk
from tkinter import filedialog, scrolledtext
from PIL import Image, ImageTk, ImageFilter
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from feature_extraction import extract_features  # Import the feature extraction function

# Global variables
loaded_image = None
preprocessed_img = None
extracted_features = None

# Adaptive filtering function
def apply_adaptive_filter(pil_img):
    """
    Apply adaptive filtering to enhance image quality
    """
    # Convert to numpy array for processing
    img_array = np.array(pil_img)
    
    # Apply adaptive filtering based on image characteristics
    if len(img_array.shape) == 3:  # Color image
        # Convert to grayscale for edge detection
        gray_img = pil_img.convert('L')
        edges = gray_img.filter(ImageFilter.FIND_EDGES)
        edge_array = np.array(edges)
        
        # Calculate edge intensity
        edge_intensity = np.mean(edge_array)
        
        # Apply different filters based on edge intensity
        if edge_intensity > 30:  # High detail image
            # Apply subtle sharpening
            filtered_img = pil_img.filter(ImageFilter.SHARPEN)
        else:  # Low detail or noisy image
            # Apply mild smoothing
            filtered_img = pil_img.filter(ImageFilter.SMOOTH)
    else:  # Grayscale image
        # Apply median filter for noise reduction
        filtered_img = pil_img.filter(ImageFilter.MedianFilter(size=3))
    
    return filtered_img

# Preprocessing function with adaptive filtering (without augmentation)
def preprocess_image(pil_img, display_label, status_label):
    global preprocessed_img
    if pil_img is None:
        status_label.config(text="❌ No image loaded!")
        return None

    # Resize to 224x224 for model input
    img_resized = pil_img.resize((224, 224))
    
    # Apply adaptive filtering based on image content
    img_filtered = apply_adaptive_filter(img_resized)

    # Convert to array & normalize
    img_array = img_to_array(img_filtered) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # (1,224,224,3)

    # No augmentation - use the filtered image directly
    preprocessed_img = img_array

    # Show the preprocessed image in Tkinter
    # Use the filtered PIL Image directly for display
    display_img = img_filtered.copy()
    
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
        # Extract features with formatted output
        formatted_output, features, features_shape = extract_features(preprocessed_img)
        extracted_features = features
        
        # Display formatted feature information
        features_text.config(state=tk.NORMAL)
        features_text.delete(1.0, tk.END)
        features_text.insert(tk.END, formatted_output)
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
        self.features_text = scrolledtext.ScrolledText(self.features_frame, height=15, state=tk.DISABLED)
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