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

# Configure TensorFlow for memory efficiency (Render Free Tier Optimization)
import tensorflow as tf

# Set environment variables for memory optimization
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

# Configure TensorFlow memory settings
try:
    # Disable GPU (Render free tier doesn't have GPU)
    tf.config.set_visible_devices([], 'GPU')
    print("GPU disabled for CPU-only deployment")
except:
    pass

# Configure threading for minimal memory usage
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')

# Configure logging
log_level = os.getenv('LOG_LEVEL', 'INFO')
logging.basicConfig(level=getattr(logging, log_level))
logger = logging.getLogger(__name__)

# Global variables for model
interpreter = None
labels = None
id_to_label = None
IMG_SIZE = None

# Model paths from environment
MODEL_PATH = os.getenv('MODEL_PATH', 'v7_plus_distracted_driver.tflite')
LABELS_PATH = os.getenv('LABELS_PATH', 'labels.pkl')

# ---------------- LOAD TFLITE MODEL ----------------
def load_tflite_model():
    """Load the TensorFlow Lite model and labels."""
    global interpreter, labels, id_to_label, IMG_SIZE
    
    logger.info("Loading TFLite model and labels...")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Files in directory: {os.listdir('.')}")
    logger.info(f"Model file path: {MODEL_PATH}")
    logger.info(f"Absolute model path: {os.path.abspath(MODEL_PATH)}")
    
    # Load labels
    try:
        with open(LABELS_PATH, 'rb') as f:
            labels = pickle.load(f)
        id_to_label = {v: k for k, v in labels.items()}
        logger.info(f"Labels loaded: {list(labels.keys())}")
    except Exception as e:
        logger.error(f"Error loading labels: {e}")
        return False
    
    # Check if model file exists
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model file not found: {MODEL_PATH}")
        return False
    
    logger.info("Model file found: v7_plus_distracted_driver.tflite")
    
    # Load TFLite model
    try:
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        logger.info(f"TFLite model loaded successfully!")
        logger.info(f"Input details: {input_details}")
        logger.info(f"Output details: {output_details}")
        
        # Set image size from input details
        input_shape = input_details[0]['shape']
        IMG_SIZE = (input_shape[1], input_shape[2]) if len(input_shape) >= 3 else (224, 224)
        logger.info(f"Image size set to: {IMG_SIZE}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error loading TFLite model: {e}")
        return False

# Load model at startup
model_loaded = load_tflite_model()

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
    """Preprocess image for TFLite model prediction"""
    try:
        img = img.convert("RGB")
        img = img.resize(IMG_SIZE)
        arr = np.array(img).astype("float32")
        arr = np.expand_dims(arr, axis=0)
        # For TFLite, we typically need to normalize to [0, 1] or use the same preprocessing as original
        arr = arr / 255.0  # Normalize to [0, 1]
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
<title>Driver Distraction AI - TFLite</title>

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

.title {
    font-size: 28px;
    font-weight: 600;
    margin-bottom: 10px;
}

.subtitle {
    font-size: 14px;
    opacity: 0.9;
    margin-bottom: 30px;
}

.upload-area {
    border: 2px dashed rgba(255,255,255,0.3);
    border-radius: 15px;
    padding: 40px 20px;
    margin: 20px 0;
    cursor: pointer;
    transition: all 0.3s ease;
    background: rgba(255,255,255,0.05);
}

.upload-area:hover {
    border-color: rgba(255,255,255,0.6);
    background: rgba(255,255,255,0.1);
}

.upload-area.dragover {
    border-color: #4CAF50;
    background: rgba(76,175,80,0.1);
}

.upload-icon {
    font-size: 48px;
    margin-bottom: 15px;
}

.upload-text {
    font-size: 16px;
    margin-bottom: 10px;
}

.upload-hint {
    font-size: 12px;
    opacity: 0.7;
}

input[type="file"] {
    display: none;
}

.btn {
    background: linear-gradient(45deg, #4CAF50, #45a049);
    color: white;
    border: none;
    padding: 12px 30px;
    border-radius: 25px;
    cursor: pointer;
    font-size: 16px;
    font-weight: 500;
    transition: all 0.3s ease;
    margin: 10px 5px;
}

.btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(76,175,80,0.4);
}

.btn:disabled {
    background: #666;
    cursor: not-allowed;
    transform: none;
    box-shadow: none;
}

.result {
    margin-top: 30px;
    padding: 20px;
    background: rgba(255,255,255,0.1);
    border-radius: 15px;
    backdrop-filter: blur(10px);
}

.prediction {
    font-size: 18px;
    font-weight: 600;
    margin: 10px 0;
}

.confidence {
    font-size: 24px;
    font-weight: 700;
    color: #4CAF50;
    margin: 10px 0;
}

.image-preview {
    max-width: 100%;
    max-height: 200px;
    border-radius: 10px;
    margin: 20px 0;
    box-shadow: 0 5px 15px rgba(0,0,0,0.3);
}

.error {
    background: rgba(244,67,54,0.2);
    color: #ff6b6b;
    padding: 15px;
    border-radius: 10px;
    margin: 20px 0;
    border: 1px solid rgba(244,67,54,0.3);
}

.success {
    background: rgba(76,175,80,0.2);
    color: #4CAF50;
    padding: 15px;
    border-radius: 10px;
    margin: 20px 0;
    border: 1px solid rgba(76,175,80,0.3);
}

.top-predictions {
    margin-top: 20px;
    text-align: left;
}

.prediction-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 12px;
    margin: 5px 0;
    background: rgba(255,255,255,0.1);
    border-radius: 8px;
    font-size: 14px;
}

.prediction-label {
    font-weight: 500;
}

.prediction-confidence {
    font-weight: 600;
    color: #4CAF50;
}

.file-info {
    font-size: 12px;
    opacity: 0.8;
    margin-top: 10px;
}

.tflite-badge {
    background: linear-gradient(45deg, #FF6B6B, #FF8E53);
    color: white;
    padding: 5px 15px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    margin-bottom: 20px;
    display: inline-block;
}

.loading {
    display: none;
    text-align: center;
    margin: 20px 0;
}

.spinner {
    border: 3px solid rgba(255,255,255,0.3);
    border-top: 3px solid white;
    border-radius: 50%;
    width: 30px;
    height: 30px;
    animation: spin 1s linear infinite;
    margin: 0 auto 10px;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}
</style>
</head>
<body>
<div class="card">
    <div class="tflite-badge">🚀 TFLite Optimized</div>
    <h1 class="title">🚗 Driver Distraction AI</h1>
    <p class="subtitle">TensorFlow Lite - Ultra Lightweight & Fast</p>
    
    <div class="upload-area" id="uploadArea">
        <div class="upload-icon">📸</div>
        <div class="upload-text">Click to upload or drag & drop</div>
        <div class="upload-hint">JPG, PNG, GIF (Max 10MB)</div>
        <input type="file" id="fileInput" accept="image/*">
    </div>
    
    <button class="btn" id="predictBtn" disabled>🔮 Analyze Driver Behavior</button>
    <button class="btn" id="clearBtn">🗑️ Clear</button>
    
    <div class="loading" id="loading">
        <div class="spinner"></div>
        <div>Analyzing with TFLite model...</div>
    </div>
    
    {% if image %}
    <img src="data:image/jpeg;base64,{{ image }}" class="image-preview" alt="Uploaded image">
    {% endif %}
    
    {% if error %}
    <div class="error">
        <strong>Error:</strong> {{ error }}
    </div>
    {% endif %}
    
    {% if label %}
    <div class="result">
        <div class="prediction">
            <strong>Prediction:</strong> {{ detail }}
        </div>
        <div class="confidence">
            Confidence: {{ conf }}%
        </div>
        
        {% if top_predictions %}
        <div class="top-predictions">
            <strong>Top 3 Predictions:</strong>
            {% for pred in top_predictions %}
            <div class="prediction-item">
                <span class="prediction-label">{{ pred.detail }}</span>
                <span class="prediction-confidence">{{ pred.confidence }}%</span>
            </div>
            {% endfor %}
        </div>
        {% endif %}
        
        {% if file_info %}
        <div class="file-info">
            📊 {{ file_info }}
        </div>
        {% endif %}
    </div>
    {% endif %}
</div>

<script>
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const predictBtn = document.getElementById('predictBtn');
const clearBtn = document.getElementById('clearBtn');
const loading = document.getElementById('loading');

// Upload area click
uploadArea.addEventListener('click', () => fileInput.click());

// Drag and drop
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        fileInput.files = files;
        handleFileSelect();
    }
});

// File selection
fileInput.addEventListener('change', handleFileSelect);

function handleFileSelect() {
    const file = fileInput.files[0];
    if (file) {
        predictBtn.disabled = false;
        uploadArea.innerHTML = `
            <div class="upload-icon">✅</div>
            <div class="upload-text">${file.name}</div>
            <div class="upload-hint">${(file.size / 1024 / 1024).toFixed(2)} MB</div>
        `;
    }
}

// Predict button
predictBtn.addEventListener('click', () => {
    const file = fileInput.files[0];
    if (!file) return;
    
    const formData = new FormData();
    formData.append('file', file);
    
    loading.style.display = 'block';
    predictBtn.disabled = true;
    
    fetch('/', {
        method: 'POST',
        body: formData
    })
    .then(response => response.text())
    .then(html => {
        document.documentElement.innerHTML = html;
    })
    .catch(error => {
        console.error('Error:', error);
        loading.style.display = 'none';
        predictBtn.disabled = false;
    });
});

// Clear button
clearBtn.addEventListener('click', () => {
    fileInput.value = '';
    predictBtn.disabled = true;
    uploadArea.innerHTML = `
        <div class="upload-icon">📸</div>
        <div class="upload-text">Click to upload or drag & drop</div>
        <div class="upload-hint">JPG, PNG, GIF (Max 10MB)</div>
    `;
    loading.style.display = 'none';
});
</script>
</body>
</html>
"""

# ---------------- ROUTES ----------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template_string(HTML)
    
    if request.method == "POST":
        start_time = datetime.now()
        label = None
        detail = None
        conf = None
        image = None
        error = None
        top_predictions = None
        file_info = None
        
        try:
            if 'file' not in request.files:
                error = "No file part in request"
                return render_template_string(HTML, error=error)
            
            file = request.files['file']
            file_size = file.content_length if hasattr(file, 'content_length') else 0
            
            # Validate file
            is_valid, message = validate_image(file, file_size)
            if not is_valid:
                error = message
                return render_template_string(HTML, error=error)
            
            # Process image
            img = Image.open(file)
            
            # Ensure image is in RGB mode (JPEG doesn't support alpha/RGBA)
            img = img.convert("RGB")
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            image = base64.b64encode(buffered.getvalue()).decode()
            
            # Check if model is loaded
            if interpreter is None:
                logger.error("TFLite model not available")
                return jsonify({'error': 'Model not loaded. Please check server logs.'}), 500
            
            logger.info("Making TFLite prediction...")
            
            # Preprocess and predict with TFLite
            x = preprocess(img)
            
            # Get input and output details
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            # Set input tensor
            interpreter.set_tensor(input_details[0]['index'], x)
            
            # Run inference
            interpreter.invoke()
            
            # Get output tensor
            preds = interpreter.get_tensor(output_details[0]['index'])
            
            # Clean up memory immediately after prediction
            import gc
            del x
            gc.collect()
            
            # Get top prediction
            idx = int(np.argmax(preds))
            label = id_to_label[idx]
            detail = CLASS_DETAILS[label]
            conf = round(float(np.max(preds)) * 100, 1)
            
            # Get top 3 predictions
            top_predictions = get_top_predictions(preds, top_k=3)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            file_info = f"Processed in {processing_time:.2f}s | {(file_size / 1024 / 1024):.2f} MB"
            
            logger.info(f"Prediction completed: {detail} ({conf}%)")
            
        except Exception as e:
            error = f"Error processing image: {str(e)}"
            logger.error(f"Error in prediction: {e}")

        return render_template_string(
            HTML, label=label, detail=detail, conf=conf, image=image,
            error=error, top_predictions=top_predictions, file_info=file_info
        )

@app.route("/debug", methods=["GET"])
def debug_info():
    """Debug endpoint to check model status"""
    return jsonify({
        'model_loaded': model_loaded,
        'interpreter_is_none': interpreter is None,
        'labels_loaded': labels is not None,
        'model_file_exists': os.path.exists(MODEL_PATH),
        'model_file_size': os.path.getsize(MODEL_PATH) if os.path.exists(MODEL_PATH) else None,
        'img_size': IMG_SIZE,
        'cwd': os.getcwd(),
        'files': os.listdir('.'),
        'tensorflow_version': tf.__version__,
        'model_type': 'TensorFlow Lite'
    })

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded and interpreter is not None,
        'model_type': 'TensorFlow Lite',
        'timestamp': datetime.now().isoformat()
    })

@app.route("/model_info", methods=["GET"])
def model_info():
    """Get model information"""
    if not model_loaded or interpreter is None:
        return jsonify({'error': 'TFLite model not loaded'}), 503
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    return jsonify({
        'model_type': 'TensorFlow Lite',
        'input_shape': input_details[0]['shape'].tolist() if hasattr(input_details[0]['shape'], 'tolist') else input_details[0]['shape'],
        'output_shape': output_details[0]['shape'].tolist() if hasattr(output_details[0]['shape'], 'tolist') else output_details[0]['shape'],
        'input_dtype': str(input_details[0]['dtype']),
        'output_dtype': str(output_details[0]['dtype']),
        'labels': list(labels.keys()) if labels else [],
        'img_size': IMG_SIZE
    })

if __name__ == "__main__":
    port = int(os.getenv('PORT', 5000))
    host = os.getenv('HOST', '0.0.0.0')
    app.run(host=host, port=port, debug=False)
