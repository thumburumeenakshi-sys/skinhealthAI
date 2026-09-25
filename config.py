import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Base Directories

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
UPLOADS_DIR = os.path.join(BASE_DIR, "static", "uploads")
DATABASE_DIR = os.path.join(BASE_DIR, "database")
DATABASE_PATH = os.path.join(DATABASE_DIR, "history.db")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

# Create directories if they do not exist
for folder in [MODELS_DIR, UPLOADS_DIR, DATABASE_DIR, DATASET_DIR]:
    os.makedirs(folder, exist_ok=True)

# Supabase Configuration (Optional Cloud History Storage)
SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_KEY = os.getenv("SUPABASE_KEY", os.getenv("SUPABASE_ANON_KEY", "")).strip()
USE_SUPABASE = bool(SUPABASE_URL and SUPABASE_KEY)

# Model Settings
MODEL_FILENAME = "mobilenetv2_skinhealth.h5"

MODEL_PATH = os.path.join(MODELS_DIR, MODEL_FILENAME)
METRICS_PATH = os.path.join(MODELS_DIR, "evaluation_results.json")
CLASS_INDICES_PATH = os.path.join(MODELS_DIR, "class_indices.json")

# Image Preprocessing Settings
IMAGE_SIZE = (224, 224)
INPUT_SHAPE = (224, 224, 3)
MAX_FILE_SIZE_MB = 10
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
MIN_IMAGE_RESOLUTION = (100, 100)
BLUR_VARIANCE_THRESHOLD = 50.0  # Laplacian variance threshold for blur detection

# Low-Confidence Prediction Threshold
CONFIDENCE_THRESHOLD = 0.50  # 50% threshold for warning flag

# 8 Skin Condition Classes & Medical Guidance Information
SKIN_CONDITIONS = {
    "AKIEC": {
        "code": "AKIEC",
        "name": "Actinic Keratosis / Intraepithelial Carcinoma",
        "category": "Precancerous / Early Malignant",
        "risk_level": "Moderate-High",
        "description": "Rough, scaly patch on sun-damaged skin caused by long-term exposure to ultraviolet (UV) radiation. Potential precursor to squamous cell carcinoma.",
        "guidance": "Avoid direct sunlight and apply broad-spectrum SPF 50+ sunscreen. Do not pick or scratch scaly lesions.",
        "dermatologist_recommendation": "Prompt consultation recommended for clinical evaluation, skin biopsy, or topical/cryotherapy treatment."
    },
    "BCC": {
        "code": "BCC",
        "name": "Basal Cell Carcinoma",
        "category": "Malignant Skin Cancer",
        "risk_level": "High",
        "description": "The most common form of skin cancer. Usually presents as a pearly bump, translucent growth, or non-healing sore. Rarely metastasizes but can cause local tissue invasion.",
        "guidance": "Protect lesion from friction or trauma. Keep clean and dry. Avoid sun exposure.",
        "dermatologist_recommendation": "Consult a dermatologist urgently for biopsy and definitive surgical or procedural treatment options."
    },
    "BKL": {
        "code": "BKL",
        "name": "Benign Keratosis (Seborrheic Keratosis / Solar Lentigo)",
        "category": "Benign Growth",
        "risk_level": "Low",
        "description": "Common non-cancerous skin growth that appears waxy, scaly, or slightly elevated (seborrheic keratosis or sun spots).",
        "guidance": "Generally harmless. Avoid aggressive scratching to prevent secondary skin irritation or infection.",
        "dermatologist_recommendation": "Routine skin check advisable. Consult if lesion rapidly changes color, size, shape, or bleeds."
    },
    "DF": {
        "code": "DF",
        "name": "Dermatofibroma",
        "category": "Benign Fibrous Nodule",
        "risk_level": "Low",
        "description": "Harmless, firm red-to-brown skin papule or nodule often found on lower extremities, commonly secondary to minor insect bites or trauma.",
        "guidance": "No immediate treatment required. Monitor for changes in appearance.",
        "dermatologist_recommendation": "Consult a dermatologist if painful, growing rapidly, or causing cosmetic concern."
    },
    "MEL": {
        "code": "MEL",
        "name": "Melanoma",
        "category": "Malignant Skin Cancer (High Priority)",
        "risk_level": "Critical / High",
        "description": "The most serious type of skin cancer originating in melanocytes. Can rapidly metastasize if not detected early. Follows the ABCDE criteria (Asymmetry, Border, Color, Diameter, Evolving).",
        "guidance": "Do not touch, rub, or attempt home remedies. Seek medical advice immediately.",
        "dermatologist_recommendation": "Urgently schedule an immediate consultation with a certified dermatologist for full-body skin examination and biopsy."
    },
    "NV": {
        "code": "NV",
        "name": "Melanocytic Nevus (Common Mole)",
        "category": "Benign Mole",
        "risk_level": "Low",
        "description": "Standard benign mole resulting from localized melanocyte accumulation. Common across all skin types.",
        "guidance": "Practice regular self-examinations. Monitor for changes using ABCDE guidelines and practice sun safety.",
        "dermatologist_recommendation": "Perform annual skin checks. Consult a dermatologist if the mole changes in shape, border, or color."
    },
    "VASC": {
        "code": "VASC",
        "name": "Vascular Lesion (Angioma / Hemangioma / Telangiectasia)",
        "category": "Benign Vascular Growth",
        "risk_level": "Low",
        "description": "Benign lesion formed by blood vessels or vascular proliferation, such as cherry angiomas or pyogenic granulomas.",
        "guidance": "Avoid traumatizing the lesion as vascular lesions bleed easily.",
        "dermatologist_recommendation": "Consult a doctor if lesion bleeds frequently, enlarges, or becomes painful."
    },
    "SCC": {
        "code": "SCC",
        "name": "Squamous Cell Carcinoma / Inflammatory Lesion",
        "category": "Malignant / Inflammatory",
        "risk_level": "High",
        "description": "Second most common skin cancer arising in squamous cells, or severe inflammatory skin condition presenting with thick red, crusted patches.",
        "guidance": "Keep lesion clean, dry, and protected from environmental irritants and UV light.",
        "dermatologist_recommendation": "Schedule professional clinical examination and biopsy to differentiate between inflammatory and neoplastic conditions."
    }
}
