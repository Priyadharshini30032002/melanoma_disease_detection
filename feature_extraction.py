# feature_extraction.py
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model

def extract_features(image_array):
    """
    Extract features from image using VGG16 pretrained model
    """
    # Load pre-trained VGG16 model without the top classification layers
    base_model = VGG16(weights='imagenet', include_top=False)
    
    # Create a model that outputs the features from the last convolutional layer
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_conv3').output)
    
    # Preprocess the image for VGG16
    if image_array.max() <= 1.0:  # If normalized to [0,1]
        image_array = image_array * 255.0
    
    # Extract features
    features = model.predict(image_array)
    
    # Flatten the features
    flattened_features = features.flatten()
    
    return flattened_features, features.shape