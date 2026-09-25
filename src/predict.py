import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.preprocessing import validate_image_file, preprocess_image_opencv
from src.model import load_trained_model, load_class_indices

# Lazy Loaders
_MODEL_INSTANCE = None
_ONNX_SESSION = None

def get_prediction_engine():
    """
    Returns an active inference engine (ONNX Session if available, otherwise TensorFlow model).
    """
    global _ONNX_SESSION, _MODEL_INSTANCE
    
    onnx_path = os.path.join(config.MODELS_DIR, "mobilenetv2_skinhealth.onnx")
    
    # Try loading ONNX Runtime first (Lightweight & Vercel Serverless Ready)
    if os.path.exists(onnx_path):
        try:
            import onnxruntime as ort
            if _ONNX_SESSION is None:
                _ONNX_SESSION = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
            input_name = _ONNX_SESSION.get_inputs()[0].name
            return ("onnx", _ONNX_SESSION, input_name)
        except Exception as e:
            print(f"ONNX Session load warning: {e}")

    # Fallback to TensorFlow Keras Model
    if _MODEL_INSTANCE is None:
        _MODEL_INSTANCE = load_trained_model(config.MODEL_PATH)
        if _MODEL_INSTANCE is None:
            from src.model import build_mobilenetv2_model
            _MODEL_INSTANCE = build_mobilenetv2_model(num_classes=len(config.SKIN_CONDITIONS))
            
    return ("tf", _MODEL_INSTANCE, None)

def predict_skin_lesion(image_path, save_processed_copy=True):
    """
    Complete Prediction Pipeline (Supports both ONNX Runtime and TensorFlow):
    1. Validate image resolution, size, and Laplacian blur.
    2. Apply OpenCV CLAHE contrast enhancement & pixel scaling.
    3. Run forward pass through ONNX / TensorFlow model.
    4. Compute ranked Softmax class probabilities across all 8 classes.
    5. Handle low-confidence thresholding (< 0.50).
    """
    # Step 1: Validate image
    validation_results = validate_image_file(image_path)
    if not validation_results["valid"]:
        return {
            "success": False,
            "errors": validation_results["errors"],
            "validation_details": validation_results["details"]
        }

    # Step 2: OpenCV Preprocessing
    try:
        input_tensor, processed_rgb = preprocess_image_opencv(image_path)
    except Exception as e:
        return {
            "success": False,
            "errors": [f"Image preprocessing failed: {str(e)}"]
        }

    # Step 3: Run Inference (ONNX or TensorFlow)
    engine_type, engine_obj, input_name = get_prediction_engine()
    
    if engine_type == "onnx":
        # ONNX Runtime forward pass
        raw_outputs = engine_obj.run(None, {input_name: input_tensor.astype(np.float32)})
        predictions = raw_outputs[0][0]
    else:
        # TensorFlow Keras forward pass
        predictions = engine_obj.predict(input_tensor, verbose=0)[0]

    # Step 4: Map predictions to class codes
    idx_to_class = load_class_indices()
    
    ranked_conditions = []
    for idx, prob in enumerate(predictions):
        class_code = idx_to_class.get(str(idx), idx_to_class.get(idx, list(config.SKIN_CONDITIONS.keys())[idx]))
        condition_info = config.SKIN_CONDITIONS.get(class_code, {
            "code": class_code,
            "name": class_code,
            "risk_level": "Unknown",
            "description": "Skin condition lesion",
            "guidance": "Consult a healthcare professional.",
            "dermatologist_recommendation": "Clinical evaluation recommended."
        })
        
        confidence_pct = round(float(prob) * 100, 2)
        ranked_conditions.append({
            "code": class_code,
            "name": condition_info["name"],
            "category": condition_info.get("category", "General"),
            "risk_level": condition_info["risk_level"],
            "confidence": confidence_pct,
            "raw_prob": float(prob),
            "description": condition_info["description"],
            "guidance": condition_info["guidance"],
            "dermatologist_recommendation": condition_info["dermatologist_recommendation"]
        })

    # Sort descending by confidence
    ranked_conditions.sort(key=lambda x: x["confidence"], reverse=True)

    top_prediction = ranked_conditions[0]
    top_confidence = top_prediction["confidence"]
    top_confidence_ratio = top_prediction["raw_prob"]

    # Step 5: Check Low Confidence Threshold
    is_low_confidence = top_confidence_ratio < config.CONFIDENCE_THRESHOLD

    low_confidence_notice = None
    if is_low_confidence:
        low_confidence_notice = (
            f"Low-confidence prediction ({top_confidence}%). Image blur, lighting variations, or rare skin feature "
            "presentations may affect model accuracy. Please consult a qualified dermatologist for a definitive clinical evaluation."
        )

    # Medical Disclaimer
    disclaimer = (
        "This tool provides AI-assisted screening information only. It is NOT a medical diagnosis "
        "and MUST NOT replace evaluation, diagnosis, or treatment by a qualified dermatologist or healthcare professional."
    )

    return {
        "success": True,
        "image_name": os.path.basename(image_path),
        "engine_used": engine_type.upper(),
        "validation_warnings": validation_results.get("warnings", []),
        "validation_details": validation_results["details"],
        "top_prediction": top_prediction,
        "is_low_confidence": is_low_confidence,
        "low_confidence_notice": low_confidence_notice,
        "ranked_conditions": ranked_conditions,
        "disclaimer": disclaimer,
        "threshold_used": config.CONFIDENCE_THRESHOLD * 100
    }
