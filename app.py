from flask import Flask, request, render_template_string, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import pickle
import os
import logging
from datetime import datetime
import base64
from io import BytesIO
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')

# Configure logging
log_level = os.getenv('LOG_LEVEL', 'INFO')
logging.basicConfig(level=getattr(logging, log_level))
logger = logging.getLogger(__name__)

# Global variables for model
model = None
labels = None
id_to_label = None
IMG_SIZE = None

# Model paths from environment
MODEL_PATH = os.getenv('MODEL_PATH', 'v7_plus_distracted_driver.keras')
LABELS_PATH = os.getenv('LABELS_PATH', 'labels.pkl')

# ---------------- LOAD MODEL ----------------
def load_model():
    """Load the TensorFlow model and labels"""
    global model, labels, id_to_label, IMG_SIZE
    
    try:
        # Debug: Check current directory and files
        current_dir = os.getcwd()
        logger.info(f"Current working directory: {current_dir}")
        logger.info(f"Files in directory: {os.listdir(current_dir)}")
        
        # Check if model file exists
        if not os.path.exists(MODEL_PATH):
            logger.error(f"Model file not found: {MODEL_PATH}")
            logger.error(f"Full path attempted: {os.path.abspath(MODEL_PATH)}")
            return False
        else:
            logger.info(f"Model file found: {MODEL_PATH}")
            
        # Load model
        model = tf.keras.models.load_model(MODEL_PATH)
        logger.info("Model loaded successfully")
        
        # Load labels
        with open(LABELS_PATH, "rb") as f:
            labels = pickle.load(f)
        
        id_to_label = {v: k for k, v in labels.items()}
        IMG_SIZE = model.input_shape[1:3]
        
        logger.info(f"Model input shape: {model.input_shape}")
        logger.info(f"Labels loaded: {list(labels.keys())}")
        return True
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        try:
            logger.info("Trying with custom_objects...")
            model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            
            with open(LABELS_PATH, "rb") as f:
                labels = pickle.load(f)
            
            id_to_label = {v: k for k, v in labels.items()}
            IMG_SIZE = model.input_shape[1:3]
            
            logger.info("Model loaded successfully with custom_objects")
            return True
        except Exception as e2:
            logger.error(f"Failed to load model with custom_objects: {e2}")
            return False

# Load model at startup
model_loaded = load_model()

CLASS_DETAILS = {
    "c0": "Safe driving",
    "c1": "Texting – Right hand",
    "c2": "Talking on phone – Right hand",
    "c3": "Texting – Left hand",
    "c4": "Talking on phone – Left hand",
    "c5": "Operating the radio",
    "c6": "Drinking",
    "c7": "Reaching behind",
    "c8": "Hair & makeup",
    "c9": "Talking to passenger"
}

# ---------------- UTILITY FUNCTIONS ----------------
def validate_image(file, file_size=None):
    """Validate uploaded image file"""
    if not file or file.filename == '':
        return False, "No file selected"
    
    # Check file extension
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    if file_ext not in allowed_extensions:
        return False, f"File type {file_ext} not allowed. Use: {', '.join(allowed_extensions)}"
    
    # Check file size (max 10MB)
    if file_size is None:
        file.seek(0, 2)  # Seek to end
        file_size = file.tell()
        file.seek(0)  # Reset to beginning
    
    if file_size > 10 * 1024 * 1024:  # 10MB
        return False, "File too large. Maximum size is 10MB"
    
    return True, "Valid file"

def get_top_predictions(preds, top_k=3):
    """Get top k predictions with their probabilities"""
    top_indices = np.argsort(preds[0])[-top_k:][::-1]
    predictions = []
    
    for idx in top_indices:
        label = id_to_label[idx]
        detail = CLASS_DETAILS[label]
        confidence = float(preds[0][idx]) * 100
        predictions.append({
            'label': label,
            'detail': detail,
            'confidence': round(confidence, 1)
        })
    
    return predictions

# ---------------- PREPROCESS ----------------
def preprocess(img):
    """Preprocess image for model prediction"""
    try:
        img = img.convert("RGB")
        img = img.resize(IMG_SIZE)
        arr = np.array(img).astype("float32")
        arr = np.expand_dims(arr, axis=0)
        # Use EfficientNet preprocessing (scales inputs appropriately)
        arr = tf.keras.applications.efficientnet.preprocess_input(arr)
        return arr
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise

# ---------------- UI ----------------
HTML = """
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Driver Distraction AI</title>

<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

* { box-sizing: border-box; }

body {
    margin: 0;
    font-family: 'Poppins', sans-serif;
    min-height: 100vh;
    background: linear-gradient(120deg, #667eea, #764ba2, #6b8dd6);
    background-size: 400% 400%;
    animation: bg 12s ease infinite;
    display: flex;
    align-items: center;
    justify-content: center;
}

@keyframes bg {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.card {
    width: 420px;
    padding: 30px;
    border-radius: 20px;
    backdrop-filter: blur(18px);
    background: rgba(255,255,255,0.18);
    box-shadow: 0 25px 45px rgba(0,0,0,0.25);
    color: white;
    text-align: center;
}

h1 {
    margin-bottom: 5px;
    font-weight: 600;
}

.subtitle {
    font-size: 14px;
    opacity: 0.85;
}

input[type=file] {
    margin: 20px 0;
    color: white;
}

button {
    background: linear-gradient(135deg, #38ef7d, #11998e);
    border: none;
    padding: 12px 22px;
    color: white;
    border-radius: 12px;
    font-size: 16px;
    cursor: pointer;
    transition: transform 0.2s ease;
}

button:hover {
    transform: scale(1.05);
}

.preview img {
    margin-top: 15px;
    width: 100%;
    border-radius: 12px;
}

.result {
    margin-top: 25px;
}

.confidence {
    width: 120px;
    height: 120px;
    border-radius: 50%;
    margin: 15px auto;
    background: conic-gradient(#38ef7d {{conf}}%, rgba(255,255,255,0.2) 0);
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 600;
}

.confidence span {
    background: rgba(0,0,0,0.6);
    padding: 8px 14px;
    border-radius: 20px;
}

.label {
    font-size: 20px;
    font-weight: 600;
    margin-top: 10px;
}

.desc {
    font-size: 14px;
    opacity: 0.9;
}

.error {
    background: rgba(220, 53, 69, 0.2);
    border: 1px solid rgba(220, 53, 69, 0.5);
    border-radius: 8px;
    padding: 10px;
    margin: 10px 0;
    font-size: 12px;
    color: #ff6b6b;
}

.loading {
    display: none;
    text-align: center;
    margin: 20px 0;
}

.loading-spinner {
    border: 3px solid rgba(255,255,255,0.3);
    border-radius: 50%;
    border-top: 3px solid #38ef7d;
    width: 30px;
    height: 30px;
    animation: spin 1s linear infinite;
    margin: 0 auto 10px;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

.top-predictions {
    margin-top: 20px;
    text-align: left;
}

.prediction-item {
    background: rgba(255,255,255,0.1);
    border-radius: 8px;
    padding: 8px 12px;
    margin: 5px 0;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.prediction-label {
    font-size: 12px;
    opacity: 0.9;
}

.prediction-confidence {
    font-weight: 600;
    font-size: 12px;
}

.file-info {
    font-size: 11px;
    opacity: 0.7;
    margin-top: 5px;
}
</style>
</head>

<body>
<div class="card">
    <h1>🚗 Driver Distraction AI</h1>
    <p class="subtitle">Deep Learning based driver behavior detection</p>

    <form method="post" enctype="multipart/form-data" id="uploadForm">
        <input type="file" name="file" id="fileInput" required accept=".jpg,.jpeg,.png,.bmp,.gif">
        <br>
        <button type="submit" id="analyzeBtn">Analyze Image</button>
    </form>
    
    <div class="loading" id="loading">
        <div class="loading-spinner"></div>
        <p>Analyzing image...</p>
    </div>
    
    {% if error %}
    <div class="error">
        {{error}}
    </div>
    {% endif %}

    {% if image %}
    <div class="preview">
        <img src="data:image/jpeg;base64,{{image}}">
    </div>
    {% endif %}

    {% if label %}
    <div class="result">
        <div class="confidence">
            <span>{{conf}}%</span>
        </div>
        <div class="label">{{detail}}</div>
        <div class="desc">Predicted Class: {{label}}</div>
        
        {% if top_predictions %}
        <div class="top-predictions">
            <div style="font-size: 14px; margin-bottom: 8px; opacity: 0.8;">Top Predictions:</div>
            {% for pred in top_predictions %}
            <div class="prediction-item">
                <span class="prediction-label">{{pred.detail}}</span>
                <span class="prediction-confidence">{{pred.confidence}}%</span>
            </div>
            {% endfor %}
        </div>
        {% endif %}
    </div>
    {% endif %}
    
    {% if file_info %}
    <div class="file-info">
        File: {{file_info.name}} | Size: {{file_info.size}} | Processed: {{file_info.processed_time}}
    </div>
    {% endif %}
</div>

<script>
document.getElementById('uploadForm').addEventListener('submit', function(e) {
    const fileInput = document.getElementById('fileInput');
    const loading = document.getElementById('loading');
    const analyzeBtn = document.getElementById('analyzeBtn');
    
    if (fileInput.files.length > 0) {
        loading.style.display = 'block';
        analyzeBtn.disabled = true;
        analyzeBtn.textContent = 'Analyzing...';
    }
});

// File input change handler
document.getElementById('fileInput').addEventListener('change', function(e) {
    const file = e.target.files[0];
    if (file) {
        const fileSize = (file.size / 1024 / 1024).toFixed(2);
        console.log(`Selected file: ${file.name} (${fileSize} MB)`);
    }
});
</script>
</body>
</html>
"""

# ---------------- ROUTES ----------------
@app.route("/", methods=["GET", "POST"])
def index():
    label = detail = conf = image = error = None
    top_predictions = None
    file_info = None

    if request.method == "POST":
        try:
            # Validate file
            file = request.files["file"]
            
            # Get file size before validation (since validation changes file pointer)
            file.seek(0, 2)  # Seek to end
            file_size = file.tell()
            file.seek(0)  # Reset to beginning
            
            is_valid, validation_msg = validate_image(file, file_size)
            
            if not is_valid:
                error = validation_msg
                return render_template_string(
                    HTML, label=label, detail=detail, conf=conf, image=image,
                    error=error, top_predictions=top_predictions, file_info=file_info
                )
            
            # Check if model is loaded
            if not model_loaded or model is None:
                error = "Model not loaded. Please check server logs."
                logger.error("Prediction attempted but model not loaded")
                return render_template_string(
                    HTML, label=label, detail=detail, conf=conf, image=image,
                    error=error, top_predictions=top_predictions, file_info=file_info
                )
            
            # Store file info before processing
            start_time = datetime.now()
            file_size_mb = file_size / 1024 / 1024  # Convert to MB
            file_info = {
                'name': file.filename,
                'size': f"{file_size_mb:.2f} MB",
                'processed_time': start_time.strftime('%H:%M:%S')
            }
            
            # Process image
            img = Image.open(file)
            
            # Ensure image is in RGB mode (JPEG doesn't support alpha/RGBA)
            img = img.convert("RGB")
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            image = base64.b64encode(buffered.getvalue()).decode()

            # Preprocess and predict
            x = preprocess(img)
            preds = model.predict(x, verbose=0)
            
            # Get top prediction
            idx = int(np.argmax(preds))
            label = id_to_label[idx]
            detail = CLASS_DETAILS[label]
            conf = round(float(np.max(preds)) * 100, 1)
            
            # Get top 3 predictions
            top_predictions = get_top_predictions(preds, top_k=3)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Prediction completed in {processing_time:.2f}s: {label} ({conf}%)")
            
        except Exception as e:
            error = f"Error processing image: {str(e)}"
            logger.error(f"Error in prediction: {e}")

    return render_template_string(
        HTML, label=label, detail=detail, conf=conf, image=image,
        error=error, top_predictions=top_predictions, file_info=file_info
    )

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded and model is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route("/model_info", methods=["GET"])
def model_info():
    """Get model information"""
    if not model_loaded or model is None:
        return jsonify({'error': 'Model not loaded'}), 503
    
    return jsonify({
        'input_shape': model.input_shape,
        'output_shape': model.output_shape,
        'classes': list(CLASS_DETAILS.keys()),
        'class_details': CLASS_DETAILS
    })

# ---------------- RUN ----------------
if __name__ == "__main__":
    logger.info("Starting Driver Distraction AI Server...")
    
    if not model_loaded:
        logger.warning("Starting server without model loaded. Please check model files.")
    else:
        logger.info("Model loaded successfully. Server ready for predictions.")
    
    # Get configuration from environment
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_ENV', 'development') == 'development'
    
    app.run(debug=debug, host=host, port=port)
