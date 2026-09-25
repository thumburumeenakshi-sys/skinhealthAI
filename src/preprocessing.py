import os
import sys
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def validate_image_file(file_path):
    """
    Validates an uploaded image file for format, size, resolution, and basic quality.
    
    Returns:
        dict: {
            "valid": bool,
            "errors": list[str],
            "warnings": list[str],
            "details": dict
        }
    """
    results = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "details": {}
    }
    
    if not os.path.exists(file_path):
        results["valid"] = False
        results["errors"].append("Image file does not exist.")
        return results
        
    # File size check
    file_size_bytes = os.path.getsize(file_path)
    file_size_mb = file_size_bytes / (1024 * 1024)
    results["details"]["file_size_mb"] = round(file_size_mb, 2)
    
    if file_size_mb > config.MAX_FILE_SIZE_MB:
        results["valid"] = False
        results["errors"].append(f"File size ({file_size_mb:.1f} MB) exceeds maximum limit of {config.MAX_FILE_SIZE_MB} MB.")
        
    # Format extension check
    ext = file_path.split('.')[-1].lower()
    if ext not in config.ALLOWED_EXTENSIONS:
        results["valid"] = False
        results["errors"].append(f"Unsupported file format '.{ext}'. Supported formats: {', '.join(config.ALLOWED_EXTENSIONS)}")
        
    # PIL image integrity check
    try:
        with Image.open(file_path) as img:
            width, height = img.size
            img_format = img.format
            results["details"]["resolution"] = f"{width}x{height}"
            results["details"]["format"] = img_format
            
            min_w, min_h = config.MIN_IMAGE_RESOLUTION
            if width < min_w or height < min_h:
                results["valid"] = False
                results["errors"].append(f"Image resolution ({width}x{height}) is too small. Minimum required: {min_w}x{min_h}.")
    except Exception as e:
        results["valid"] = False
        results["errors"].append(f"Corrupted or invalid image file: {str(e)}")
        return results

    # OpenCV Image Quality Check (Blur, Brightness, Contrast)
    cv_img = cv2.imread(file_path)
    if cv_img is None:
        results["valid"] = False
        results["errors"].append("OpenCV failed to read image.")
        return results

    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    
    # Blur detection via Laplacian Variance
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    results["details"]["blur_variance"] = round(float(blur_score), 2)
    if blur_score < config.BLUR_VARIANCE_THRESHOLD:
        results["warnings"].append(f"Image appears blurry (Quality score: {blur_score:.1f}). This may affect AI accuracy.")

    # Brightness check
    mean_brightness = float(np.mean(gray))
    results["details"]["mean_brightness"] = round(mean_brightness, 2)
    if mean_brightness < 40:
        results["warnings"].append("Image is under-exposed (too dark). Good lighting is recommended.")
    elif mean_brightness > 220:
        results["warnings"].append("Image is over-exposed (too bright/washed out). Avoid harsh direct glare.")

    return results

def preprocess_image_opencv(image_path_or_bytes):
    """
    Preprocesses an image using OpenCV:
    1. Loads image & converts BGR to RGB
    2. Applies CLAHE contrast enhancement in LAB color space
    3. Resizes image to 224x224
    4. Normalizes pixel values for MobileNetV2 (-1 to 1)
    
    Returns:
        tuple: (preprocessed_tensor, processed_rgb_image_array)
    """
    if isinstance(image_path_or_bytes, str):
        img_bgr = cv2.imread(image_path_or_bytes)
        if img_bgr is None:
            raise ValueError(f"Could not read image from path: {image_path_or_bytes}")
    else:
        # Byte array
        np_arr = np.frombuffer(image_path_or_bytes, np.uint8)
        img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("Could not decode image from byte buffer.")

    # Convert BGR to LAB color space for adaptive contrast enhancement
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    # Apply CLAHE to L-channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    
    # Merge channels back and convert to BGR then RGB
    limg = cv2.merge((cl, a_channel, b_channel))
    enhanced_bgr = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    img_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)

    # Resize to target input dimensions (224, 224)
    resized_rgb = cv2.resize(img_rgb, config.IMAGE_SIZE, interpolation=cv2.INTER_AREA)

    # Convert to float32 array
    img_array = np.array(resized_rgb, dtype=np.float32)

    # Apply MobileNetV2 preprocessing: scales pixels from [0, 255] to [-1, 1]
    preprocessed_input = tf.keras.applications.mobilenet_v2.preprocess_input(img_array.copy())
    
    # Expand dims to batch shape (1, 224, 224, 3)
    batch_input = np.expand_dims(preprocessed_input, axis=0)

    return batch_input, resized_rgb
