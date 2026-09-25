# SkinHealth AI — AI-Assisted Skin Disease Detection Using Deep Learning

**SkinHealth AI** is a complete, full-stack web application for preliminary AI-assisted skin lesion screening. It leverages a custom MobileNetV2 deep learning architecture, OpenCV image validation & preprocessing, Flask REST backend, SQLite prediction history database, and a modern responsive web frontend UI.

> [!IMPORTANT]
> **Medical Disclaimer**: SkinHealth AI provides AI-assisted screening information only. It is **NOT** a medical diagnostic tool and **MUST NOT** replace evaluation, diagnosis, or treatment by a licensed dermatologist or healthcare professional.

---

## 🌟 Key Features

- **8 Target Skin Condition Classes**:
  1. `AKIEC` — Actinic Keratosis / Intraepithelial Carcinoma
  2. `BCC` — Basal Cell Carcinoma
  3. `BKL` — Benign Keratosis (Seborrheic Keratosis / Solar Lentigo)
  4. `DF` — Dermatofibroma
  5. `MEL` — Melanoma
  6. `NV` — Melanocytic Nevus (Common Mole)
  7. `VASC` — Vascular Lesion (Angioma / Hemangioma)
  8. `SCC` — Squamous Cell Carcinoma / Inflammatory Lesion
- **OpenCV Image Validation & Quality Assurance**:
  - Image format verification (JPEG, PNG, WEBP)
  - File size and minimum resolution checks (min 100x100)
  - Blur detection via Laplacian variance calculation
  - Brightness and contrast evaluation
- **Preprocessing Pipeline**:
  - Contrast Limited Adaptive Histogram Equalization (CLAHE) on LAB color space for lesion boundary enhancement
  - Image scaling to 224×224 pixels and MobileNetV2 normalization `[-1, 1]`
- **MobileNetV2 Deep Learning Model**:
  - Pretrained ImageNet backbone with custom classification head (GlobalAveragePooling2D, BatchNormalization, Dense 256, Dropout, Dense 128, Softmax 8)
  - Training, fine-tuning, and empirical evaluation metrics saved directly to disk (`evaluation_results.json`)
- **Low-Confidence Warning Flag**:
  - Configurable threshold (`CONFIDENCE_THRESHOLD = 0.50` / `50%`)
  - Displays prominent warnings for uncertain predictions and directs users to dermatologists
- **Prediction History (SQLite & Supabase Sync)**:
  - Supports dual database storage: local SQLite for offline privacy + optional Supabase cloud history sync via PostgREST REST API.
  - Includes standard SQL schema script (`database/supabase_schema.sql`).

- **Modern Responsive Frontend**:
  - Drag-and-drop file upload zone + Live HTML5 Camera capture modal
  - Multi-step animated progress spinner
  - Ranked condition progress bars for all 8 classes
  - Dermatology guidance & ABCDE melanoma monitoring rule

---

## 🛠️ Technology Stack

- **Language**: Python 3.11+
- **Deep Learning Framework**: TensorFlow / Keras (MobileNetV2)
- **Computer Vision & Image Processing**: OpenCV (`cv2`), PIL (Pillow)
- **Data Analysis & Evaluation**: NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn
- **Backend Web Framework**: Flask, Flask-CORS
- **Database**: SQLite3
- **Frontend UI**: HTML5, CSS3 (Modern Medical Dark Theme), JavaScript ES6+ (Fetch API, HTML5 Canvas, WebCam API)

---

## 📁 Modular Project Structure

```
SkinHealthAI/
├── app.py                      # Flask REST API server & static route handler
├── config.py                   # Centralized configuration (thresholds, paths, 8 class info)
├── requirements.txt            # Python dependencies
├── README.md                   # Complete setup and operational instructions
├── models/                     # Saved trained model artifacts
│   ├── mobilenetv2_skinhealth.h5
│   ├── class_indices.json
│   ├── evaluation_results.json
│   └── confusion_matrix.png
├── database/                   # SQLite prediction history storage
│   └── history.db
├── dataset/                    # Training, validation, and test datasets
│   ├── train/
│   ├── val/
│   └── test/
├── src/                        # Core backend Python modules
│   ├── __init__.py
│   ├── preprocessing.py        # Image quality validation, OpenCV CLAHE & MobileNet preprocessing
│   ├── model.py                # MobileNetV2 architecture builder & load functions
│   ├── train.py                # Complete training, fine-tuning, & test evaluation script
│   ├── predict.py              # Prediction & ranking inference engine
│   ├── dataset.py              # Synthetic lesion generator & dataset pipeline helper
│   └── db.py                   # SQLite database CRUD operations
├── static/                     # Frontend static assets
│   ├── css/
│   │   └── style.css           # Modern medical UI stylesheet
│   ├── js/
│   │   └── main.js             # Client-side controller, drag-drop, camera API, REST fetch
│   └── uploads/                # Temporary uploaded lesion images
└── templates/                  # Single Page Application HTML templates
    └── index.html              # Main web interface template
```

---

## 🚀 Setup & Installation Instructions

### Prerequisites
- Python 3.9, 3.10, or 3.11 installed.

### Step 1: Clone / Open Project Directory
Navigate into the project directory:
```bash
cd C:\Users\thumb\.gemini\antigravity\scratch\SkinHealthAI
```

### Step 2: Create & Activate Virtual Environment
On Windows (PowerShell):
```pwsh
python -m venv venv
.\venv\Scripts\Activate.ps1
```
On Linux / macOS:
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🎓 Model Training & Evaluation Pipeline

The project includes an automated sample dataset generator and training pipeline (`src/train.py`). If custom dataset files are not provided in `dataset/`, running the training script automatically synthesizes lesion dataset samples for all 8 classes to verify end-to-end training, accuracy calculation, and model saving.

To train and evaluate the MobileNetV2 model:
```bash
python src/train.py
```

### Training Steps Executed:
1. Generates / loads train, val, and test image split directories.
2. Applies image data augmentation (rotation, zoom, flips, shifts).
3. Fine-tunes MobileNetV2 with EarlyStopping and ReduceLROnPlateau callbacks.
4. Evaluates test set performance and produces empirical metrics:
   - Accuracy, Macro Precision, Recall, F1-Score
   - Saves confusion matrix plot to `models/confusion_matrix.png`
   - Saves metrics summary to `models/evaluation_results.json`
5. Saves trained model to `models/mobilenetv2_skinhealth.h5`.

---

## ⚡ Vercel & Cloud Deployment

### 1. Vercel Serverless Deployment (Instant 50MB Bundle)
Vercel serverless functions have a 500 MB limit. Standard `tensorflow` (~2.4 GB) exceeds this limit. 

This repository includes a pre-exported **ONNX Runtime engine** (`models/mobilenetv2_skinhealth.onnx` ~9.8 MB) and `vercel.json` configuration.

- **Requirements File**: `requirements.txt` contains `onnxruntime` + `opencv-python-headless` (~65 MB total bundle size).
- **Vercel Setup**: Connect your GitHub repository (`thumburumeenakshi-sys/skinhealthAI`) to Vercel. Vercel will automatically build and deploy the application in under 30 seconds!

---

## 💻 Running the Web Application Locally

To start the Flask local web server:
```bash
python app.py
```

Open your web browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 🔌 API Endpoints Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Serves main single-page application interface |
| `POST` | `/api/predict` | Accepts image file upload, executes OpenCV validation + MobileNetV2 inference, stores and returns prediction results |
| `GET` | `/api/history` | Returns list of saved prediction records from SQLite database |
| `POST` | `/api/history/delete` | Deletes a record or clears all history records |
| `GET` | `/api/model-info` | Returns model architecture metadata, 8 class definitions, and empirical evaluation metrics |

---

## 🛡️ Medical Disclaimer & Safety

This web application is strictly for **educational, academic, and screening demonstration purposes**. It must not be used as a primary diagnostic tool. Users are explicitly instructed in the UI to consult a licensed dermatologist or medical professional for clinical examination and biopsy.
