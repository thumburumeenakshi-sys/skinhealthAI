import os
import sys
import tensorflow as tf
import tf2onnx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.model import load_trained_model

def export_to_onnx():
    """Converts saved Keras MobileNetV2 model to ONNX format for lightweight serverless deployment."""
    print("Loading Keras MobileNetV2 model...")
    keras_model = load_trained_model(config.MODEL_PATH)
    if keras_model is None:
        raise ValueError(f"Could not load Keras model from {config.MODEL_PATH}")

    onnx_path = os.path.join(config.MODELS_DIR, "mobilenetv2_skinhealth.onnx")
    print(f"Converting Keras model to ONNX format -> {onnx_path}")
    
    spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input_1"),)
    model_proto, _ = tf2onnx.convert.from_keras(keras_model, input_signature=spec, output_path=onnx_path)
    
    file_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"ONNX Model exported successfully! File size: {file_size_mb:.2f} MB")
    return onnx_path

if __name__ == "__main__":
    export_to_onnx()
