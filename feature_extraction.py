# feature_extraction.py
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
import cv2
from skimage.feature import hog, local_binary_pattern
from skimage import exposure
from skimage.measure import regionprops
import math

def extract_melanoma_features(image_array):
    """
    Extract comprehensive features for melanoma detection following the ABCDE rule
    and using VGG16 as described in the hybrid LSTM-CNN paper
    """
    # Convert image to appropriate format
    if image_array.max() <= 1.0:
        img_uint8 = (image_array[0] * 255).astype(np.uint8)
    else:
        img_uint8 = image_array[0].astype(np.uint8)
    
    # Convert to grayscale for some features
    if len(img_uint8.shape) == 3:
        gray_img = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    else:
        gray_img = img_uint8
    
    # Extract ABCDE rule features
    abcde_features = extract_abcde_features(img_uint8, gray_img)
    
    # Extract deep features from VGG16
    deep_features = extract_vgg16_features(image_array)
    
    # Extract handcrafted features
    handcrafted_features = extract_handcrafted_features(img_uint8, gray_img)
    
    # Combine all features
    all_features = {
        'abcde_features': abcde_features,
        'deep_features': deep_features,
        'handcrafted_features': handcrafted_features
    }
    
    return all_features

def extract_abcde_features(color_img, gray_img):
    """
    Extract ABCDE rule features as described in the paper
    """
    features = {}
    
    # Segment the lesion (simplified version)
    _, binary_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        
        # A: Asymmetry
        area = cv2.contourArea(largest_contour)
        features['area'] = area
        
        # Calculate asymmetry (simplified version)
        moments = cv2.moments(largest_contour)
        if area > 0:
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])
            
            # Split image and calculate area differences
            height, width = gray_img.shape
            left_area = np.sum(binary_img[:, :cx] > 0)
            right_area = np.sum(binary_img[:, cx:] > 0)
            asymmetry = abs(left_area - right_area) / area
            features['asymmetry'] = asymmetry
        
        # B: Border Irregularity
        perimeter = cv2.arcLength(largest_contour, True)
        features['perimeter'] = perimeter
        
        # Convex hull for border irregularity
        hull = cv2.convexHull(largest_contour)
        hull_perimeter = cv2.arcLength(hull, True)
        if hull_perimeter > 0:
            border_irregularity = perimeter / hull_perimeter
            features['border_irregularity'] = border_irregularity
        
        # C: Color Variation
        if len(color_img.shape) == 3:
            color_std = np.std(color_img, axis=(0, 1))
            features['color_std_r'] = color_std[0]
            features['color_std_g'] = color_std[1]
            features['color_std_b'] = color_std[2]
        
        # D: Diameter
        if area > 0:
            diameter = 2 * math.sqrt(area / math.pi)
            features['diameter'] = diameter
        
        # E: Evolving (Roundness/Circularity)
        if perimeter > 0:
            circularity = 4 * math.pi * area / (perimeter ** 2)
            features['circularity'] = circularity
    
    return features

def extract_vgg16_features(image_array):
    """
    Extract deep features from VGG16
    """
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Preprocess the image for VGG16
    if image_array.max() <= 1.0:
        vgg_input = image_array * 255.0
    else:
        vgg_input = image_array.copy()
    
    vgg_input = preprocess_input(vgg_input)
    
    # Extract features from multiple layers
    layer_names = ['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3', 'block5_conv3']
    outputs = [base_model.get_layer(name).output for name in layer_names]
    feature_extraction_model = Model(inputs=base_model.input, outputs=outputs)
    
    # Extract features
    features = feature_extraction_model.predict(vgg_input, verbose=0)
    
    return [feat.flatten() for feat in features]

def extract_handcrafted_features(color_img, gray_img):
    """
    Extract additional handcrafted features
    """
    features = {}
    
    # Texture features using LBP
    lbp = local_binary_pattern(gray_img, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp, bins=np.arange(0, 10), density=True)
    features['lbp_texture'] = lbp_hist
    
    # HOG features
    hog_features, _ = hog(gray_img, orientations=8, pixels_per_cell=(16, 16),
                         cells_per_block=(1, 1), visualize=True)
    features['hog_features'] = hog_features
    
    # Edge features
    edges = cv2.Canny(gray_img, 100, 200)
    edge_density = np.sum(edges > 0) / edges.size
    features['edge_density'] = edge_density
    
    # Statistical features
    mean_intensity = np.mean(gray_img)
    std_intensity = np.std(gray_img)
    features['intensity_mean'] = mean_intensity
    features['intensity_std'] = std_intensity
    
    return features

def flatten_features(all_features):
    """
    Flatten all features into a single vector
    """
    flattened = []
    
    # Flatten ABCDE features
    for key, value in all_features['abcde_features'].items():
        if isinstance(value, (int, float)):
            flattened.append(value)
        elif isinstance(value, np.ndarray):
            flattened.extend(value)
    
    # Flatten deep features
    for deep_feat in all_features['deep_features']:
        flattened.extend(deep_feat)
    
    # Flatten handcrafted features
    for key, value in all_features['handcrafted_features'].items():
        if isinstance(value, (int, float)):
            flattened.append(value)
        elif isinstance(value, np.ndarray):
            flattened.extend(value)
    
    return np.array(flattened)

def format_feature_output(all_features):
    """
    Format the feature extraction results with ABCDE rule features
    """
    output_lines = []
    
    # Header
    output_lines.append("MELANOMA FEATURE EXTRACTION REPORT (ABCDE RULE)")
    output_lines.append("=" * 60)
    output_lines.append("")
    
    # ABCDE Features
    output_lines.append("ABCDE RULE FEATURES:")
    output_lines.append("-" * 30)
    abcde = all_features['abcde_features']
    for key, value in abcde.items():
        if isinstance(value, (int, float)):
            output_lines.append(f"{key:<20} {value:>10.4f}")
    
    output_lines.append("")
    
    # Deep Features Summary
    output_lines.append("DEEP FEATURES (VGG16)")
    output_lines.append("-" * 30)
    for i, feat in enumerate(all_features['deep_features']):
        layer_name = f"Layer {i+1}"
        feature_count = len(feat)
        output_lines.append(f"{layer_name:<15} {feature_count:>8} features")
    
    output_lines.append("")
    
    # Handcrafted Features Summary
    output_lines.append("HANDCRAFTED FEATURES")
    output_lines.append("-" * 30)
    handcrafted = all_features['handcrafted_features']
    
    output_lines.append("Texture Features:")
    if 'lbp_texture' in handcrafted:
        lbp_values = handcrafted['lbp_texture'][:5]
        lbp_str = " ".join([f"{v:.3f}" for v in lbp_values])
        output_lines.append(f"  LBP:         {lbp_str}")
    
    output_lines.append("Edge Features:")
    if 'edge_density' in handcrafted:
        output_lines.append(f"  Edge Density: {handcrafted['edge_density']:.4f}")
    
    output_lines.append("Intensity Features:")
    if 'intensity_mean' in handcrafted:
        output_lines.append(f"  Mean:        {handcrafted['intensity_mean']:.2f}")
        output_lines.append(f"  Std:         {handcrafted['intensity_std']:.2f}")
    
    output_lines.append("")
    
    # Total Features
    total_abcde = len(all_features['abcde_features'])
    total_deep = sum(len(feat) for feat in all_features['deep_features'])
    total_handcrafted = sum(1 if isinstance(v, (int, float)) else len(v) 
                          for v in all_features['handcrafted_features'].values())
    
    output_lines.append("SUMMARY")
    output_lines.append("-" * 30)
    output_lines.append(f"ABCDE Features:          {total_abcde:>8}")
    output_lines.append(f"Deep Features:           {total_deep:>8}")
    output_lines.append(f"Handcrafted Features:    {total_handcrafted:>5}")
    output_lines.append(f"Total Features:          {total_abcde + total_deep + total_handcrafted:>8}")
    
    return "\n".join(output_lines)

# Main function to extract features
def extract_features(image_array):
    """
    Main function to extract features (compatible with existing code)
    """
    all_features = extract_melanoma_features(image_array)
    flattened_features = flatten_features(all_features)
    formatted_output = format_feature_output(all_features)
    return formatted_output, flattened_features, all_features['deep_features'][0].shape