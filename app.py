import os
import sys
import json
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from src.predict import predict_skin_lesion
from src.db import init_db, save_prediction, get_prediction_history, delete_prediction_history, clear_all_history

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# Ensure required directories exist and DB is initialized
try:
    os.makedirs(config.UPLOADS_DIR, exist_ok=True)
    init_db()
except Exception as e:
    print(f"App setup warning: {e}")

@app.route("/")
def index():
    """Renders main application interface."""
    return render_template("index.html")

@app.route("/static/uploads/<path:filename>")
def serve_uploads(filename):
    """Serves uploaded lesion images from configured upload directory (handles local and Vercel /tmp)."""
    return send_from_directory(config.UPLOADS_DIR, filename)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    API Endpoint: Image Upload & AI Lesion Screening
    Expects multipart form data with file field 'image'.
    """
    if "image" not in request.files:
        return jsonify({"success": False, "errors": ["No image file provided in request."]}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"success": False, "errors": ["Selected file has an empty filename."]}), 400

    # Save uploaded file safely
    filename = secure_filename(file.filename)
    import time
    timestamp_prefix = int(time.time())
    saved_filename = f"{timestamp_prefix}_{filename}"
    
    os.makedirs(config.UPLOADS_DIR, exist_ok=True)
    file_path = os.path.join(config.UPLOADS_DIR, saved_filename)
    file.save(file_path)

    # Run AI Prediction Pipeline
    result = predict_skin_lesion(file_path)

    if not result["success"]:
        if os.path.exists(file_path):
            os.remove(file_path)
        return jsonify(result), 400

    # URL path for frontend rendering
    image_url = f"/static/uploads/{saved_filename}"
    result["image_url"] = image_url

    # Save to SQLite / Supabase history
    try:
        record_id = save_prediction(result, image_url=image_url)
        result["history_record_id"] = record_id
    except Exception as e:
        print(f"Warning: Failed to save to history DB: {e}")

    return jsonify(result), 200

@app.route("/api/history", methods=["GET"])
def api_get_history():
    """API Endpoint: Fetch prediction history."""
    try:
        limit = request.args.get("limit", default=50, type=int)
        history = get_prediction_history(limit=limit)
        return jsonify({"success": True, "history": history})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/history/delete", methods=["POST"])
def api_delete_history():
    """API Endpoint: Delete a history entry or clear all."""
    try:
        data = request.get_json() or {}
        record_id = data.get("record_id")
        if record_id == "all":
            clear_all_history()
            return jsonify({"success": True, "message": "All prediction history cleared."})
        elif record_id:
            delete_prediction_history(record_id)
            return jsonify({"success": True, "message": f"Record {record_id} deleted."})
        else:
            return jsonify({"success": False, "error": "Missing record_id parameter."}), 400
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/model-info", methods=["GET"])
def api_model_info():
    """API Endpoint: Return model metadata, 8 class descriptions, and actual evaluation metrics."""
    metrics = {}
    if os.path.exists(config.METRICS_PATH):
        try:
            with open(config.METRICS_PATH, "r") as f:
                metrics = json.load(f)
        except Exception as e:
            metrics = {"error": f"Failed to load metrics: {str(e)}"}

    model_trained = os.path.exists(config.MODEL_PATH) or os.path.exists(os.path.join(config.MODELS_DIR, "mobilenetv2_skinhealth.onnx"))

    return jsonify({
        "success": True,
        "model_name": "MobileNetV2 ONNX / Keras Engine",
        "input_shape": config.INPUT_SHAPE,
        "confidence_threshold": config.CONFIDENCE_THRESHOLD * 100,
        "model_trained": model_trained,
        "database_backend": "Supabase Cloud + SQLite" if config.USE_SUPABASE else "Local SQLite",
        "use_supabase": config.USE_SUPABASE,
        "num_classes": len(config.SKIN_CONDITIONS),
        "classes": config.SKIN_CONDITIONS,
        "evaluation_metrics": metrics
    })

if __name__ == "__main__":
    print(f"Starting SkinHealth AI Server on http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
