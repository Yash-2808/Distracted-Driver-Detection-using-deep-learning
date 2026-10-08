from flask import Flask, request, render_template_string, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import pickle
import os
import io
import time
import logging
import base64
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

try:
    if os.getenv('FORCE_CPU', 'false').lower() == 'true':
        tf.config.set_visible_devices([], 'GPU')
except Exception:
    pass

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'driverguard-ai-secret')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max

log_level = os.getenv('LOG_LEVEL', 'INFO')
logging.basicConfig(level=getattr(logging, log_level), format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Model state
model = None
labels = None
id_to_label = None
IMG_SIZE = (224, 224)
MODEL_PATH = os.getenv('MODEL_PATH', 'v7_plus_distracted_driver.keras')
LABELS_PATH = os.getenv('LABELS_PATH', 'labels.pkl')
model_loaded = False

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

# ---------------- MODEL LOADING ----------------
def load_model():
    """Load the trained model and labels."""
    global model, labels, id_to_label, IMG_SIZE, model_loaded
    
    logger.info("Loading DriverGuard AI model and labels...")
    
    # Load labels
    try:
        with open(LABELS_PATH, 'rb') as f:
            labels = pickle.load(f)
        id_to_label = {v: k for k, v in labels.items()}
        logger.info(f"Labels loaded: {labels}")
    except Exception as e:
        logger.error(f"Error loading labels: {e}")
        labels = {f"c{i}": i for i in range(10)}
        id_to_label = {i: f"c{i}" for i in range(10)}
    
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model file not found: {MODEL_PATH}")
        return False
        
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        input_shape = model.input_shape
        IMG_SIZE = (input_shape[1], input_shape[2]) if len(input_shape) >= 3 else (224, 224)
        logger.info(f"Model loaded successfully with input shape: {input_shape}")
        
        # Warm up model
        dummy = np.zeros((1, IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.float32)
        _ = model.predict(dummy, verbose=0)
        
        model_loaded = True
        return True
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False

# Load model at startup
model_loaded = load_model()

# ---------------- ACCURACY PREPROCESSING ----------------
def preprocess_image(img, target_size=(224, 224)):
    """
    High-accuracy preprocessing:
    1. Fix EXIF orientation (fixes rotated phone captures)
    2. Aspect-preserving resize with letterbox padding (prevents squashing driver posture)
    """
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    
    orig_w, orig_h = img.size
    target_w, target_h = target_size
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    
    resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    padded = Image.new("RGB", target_size, (20, 24, 32))
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    padded.paste(resized, (paste_x, paste_y))
    
    arr = np.array(padded, dtype=np.float32)
    return np.expand_dims(arr, axis=0), padded

def predict_driver_state(img):
    """Run model prediction and return formatted results."""
    x, padded_img = preprocess_image(img, IMG_SIZE)
    preds = model.predict(x, verbose=0)[0]
    
    # Temperature smoothing
    exp_preds = np.exp(np.log(np.clip(preds, 1e-7, 1.0)) / 0.95)
    calibrated_probs = exp_preds / np.sum(exp_preds)
    
    top_idx = int(np.argmax(calibrated_probs))
    top_label = id_to_label.get(top_idx, f"c{top_idx}")
    top_detail = CLASS_DETAILS.get(top_label, "Unknown")
    top_conf = round(float(calibrated_probs[top_idx]) * 100, 1)
    
    # Top 3 predictions
    top_indices = np.argsort(calibrated_probs)[-3:][::-1]
    top_predictions = []
    for idx in top_indices:
        lbl = id_to_label.get(idx, f"c{idx}")
        top_predictions.append({
            'label': lbl,
            'detail': CLASS_DETAILS.get(lbl, lbl),
            'confidence': round(float(calibrated_probs[idx]) * 100, 1)
        })
        
    return {
        'label': top_label,
        'detail': top_detail,
        'confidence': top_conf,
        'top_predictions': top_predictions,
        'is_safe': top_label == "c0"
    }

# ---------------- UI TEMPLATE ----------------
HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DriverGuard AI – Driver Distraction Detection</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">

<style>
* { box-sizing: border-box; margin: 0; padding: 0; }

body {
    font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
    min-height: 100vh;
    background: radial-gradient(circle at 10% 20%, #111827 0%, #080b11 90%);
    color: #f3f4f6;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 30px 16px;
}

.card {
    width: 100%;
    max-width: 480px;
    padding: 32px 28px;
    border-radius: 24px;
    background: rgba(22, 28, 42, 0.75);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    box-shadow: 0 30px 60px rgba(0,0,0,0.5), 0 0 30px rgba(0, 242, 254, 0.05);
    text-align: center;
    position: relative;
    overflow: hidden;
}

.card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, #00f2fe, #4facfe, transparent);
}

.brand-icon {
    width: 52px;
    height: 52px;
    border-radius: 14px;
    background: linear-gradient(135deg, #00f2fe, #4facfe);
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 24px;
    color: #0b0f19;
    box-shadow: 0 0 25px rgba(0, 242, 254, 0.35);
    margin-bottom: 12px;
}

h1 {
    font-size: 24px;
    font-weight: 800;
    letter-spacing: -0.5px;
    background: linear-gradient(120deg, #ffffff, #d1d5db);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 4px;
}

.subtitle {
    font-size: 13px;
    color: #9ca3af;
    margin-bottom: 22px;
}

/* Mode Switcher */
.mode-tabs {
    display: flex;
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 4px;
    margin-bottom: 20px;
    gap: 4px;
}

.mode-btn {
    flex: 1;
    background: transparent;
    border: none;
    color: #9ca3af;
    padding: 8px 12px;
    border-radius: 8px;
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 6px;
    transition: all 0.2s ease;
}

.mode-btn.active {
    background: linear-gradient(135deg, #00f2fe, #4facfe);
    color: #0b0f19;
    font-weight: 700;
    box-shadow: 0 4px 12px rgba(0, 242, 254, 0.25);
}

/* Upload Section */
.upload-box {
    border: 2px dashed rgba(255, 255, 255, 0.14);
    border-radius: 14px;
    padding: 22px 14px;
    background: rgba(255, 255, 255, 0.02);
    cursor: pointer;
    margin-bottom: 16px;
    transition: all 0.2s ease;
}

.upload-box:hover {
    border-color: #00f2fe;
    background: rgba(0, 242, 254, 0.04);
}

.upload-box i {
    font-size: 28px;
    color: #00f2fe;
    margin-bottom: 8px;
}

.upload-text {
    font-size: 13px;
    font-weight: 600;
    color: #e5e7eb;
}

.upload-sub {
    font-size: 11px;
    color: #9ca3af;
    margin-top: 2px;
}

input[type=file] {
    display: none;
}

/* Action Buttons */
.btn-primary {
    width: 100%;
    background: linear-gradient(135deg, #00f2fe, #4facfe);
    border: none;
    padding: 13px 20px;
    color: #0b0f19;
    border-radius: 12px;
    font-size: 15px;
    font-weight: 700;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    transition: all 0.2s ease;
    box-shadow: 0 4px 20px rgba(0, 242, 254, 0.3);
}

.btn-primary:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 25px rgba(0, 242, 254, 0.45);
}

.btn-danger {
    background: linear-gradient(135deg, #ff1744, #f50057);
    color: #fff;
    box-shadow: 0 4px 20px rgba(255, 23, 68, 0.3);
}

/* Live Camera Viewport */
.camera-box {
    display: none;
    margin-bottom: 16px;
    border-radius: 14px;
    overflow: hidden;
    background: #000;
    border: 1px solid rgba(255, 255, 255, 0.1);
    position: relative;
}

#webcamVideo {
    width: 100%;
    height: auto;
    display: block;
    transform: scaleX(-1);
}

.live-badge {
    position: absolute;
    top: 10px;
    left: 10px;
    background: rgba(0,0,0,0.7);
    backdrop-filter: blur(8px);
    padding: 4px 10px;
    border-radius: 20px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    display: flex;
    align-items: center;
    gap: 6px;
    border: 1px solid rgba(255, 255, 255, 0.1);
}

.live-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #ff1744;
    box-shadow: 0 0 8px #ff1744;
    animation: blink 1.2s infinite;
}

@keyframes blink {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
}

/* Preview Image */
.preview-box {
    margin-top: 18px;
    border-radius: 14px;
    overflow: hidden;
    border: 1px solid rgba(255, 255, 255, 0.1);
    background: #000;
}

.preview-box img {
    width: 100%;
    max-height: 280px;
    object-fit: contain;
    display: block;
}

/* Result Display */
.result-card {
    margin-top: 20px;
    padding: 20px;
    border-radius: 16px;
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.08);
}

.confidence-dial {
    width: 100px;
    height: 100px;
    border-radius: 50%;
    margin: 0 auto 12px;
    background: conic-gradient(var(--dial-color, #00e676) calc(var(--conf-val, 0) * 1%), rgba(255, 255, 255, 0.08) 0);
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 0 20px rgba(0, 242, 254, 0.15);
}

.confidence-dial span {
    width: 78px;
    height: 78px;
    background: #111827;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    font-size: 17px;
}

.result-label {
    font-size: 20px;
    font-weight: 800;
    margin-bottom: 4px;
    color: #f9fafb;
}

.result-class {
    font-size: 12px;
    color: #9ca3af;
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 16px;
}

.top-predictions {
    margin-top: 14px;
    text-align: left;
}

.top-predictions-title {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: #9ca3af;
    margin-bottom: 8px;
}

.pred-item {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 8px;
    padding: 8px 12px;
    margin-bottom: 6px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.pred-name {
    font-size: 12px;
    font-weight: 600;
    color: #e5e7eb;
}

.pred-conf {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    font-weight: 700;
    color: #00f2fe;
}

/* Spinner */
.loading {
    display: none;
    margin: 18px 0;
}

.spinner {
    width: 32px;
    height: 32px;
    border: 3px solid rgba(0, 242, 254, 0.2);
    border-top: 3px solid #00f2fe;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
    margin: 0 auto 8px;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

.error-box {
    background: rgba(239, 68, 68, 0.15);
    border: 1px solid rgba(239, 68, 68, 0.4);
    border-radius: 10px;
    padding: 10px;
    margin-top: 14px;
    font-size: 12px;
    color: #fca5a5;
}
</style>
</head>

<body>
<div class="card">
    <div class="brand-icon">
        <i class="fa-solid fa-shield-halved"></i>
    </div>
    <h1>DriverGuard AI</h1>
    <p class="subtitle">Deep Learning Driver Distraction Detection</p>

    <!-- Mode Selector -->
    <div class="mode-tabs">
        <button class="mode-btn active" id="btnTabUpload" onclick="setMode('upload')">
            <i class="fa-solid fa-cloud-arrow-up"></i> Upload Image
        </button>
        <button class="mode-btn" id="btnTabCamera" onclick="setMode('camera')">
            <i class="fa-solid fa-camera"></i> Live Camera
        </button>
    </div>

    <!-- Upload Mode Form -->
    <div id="uploadSection">
        <form method="post" enctype="multipart/form-data" id="uploadForm">
            <div class="upload-box" onclick="document.getElementById('fileInput').click()">
                <i class="fa-solid fa-image"></i>
                <div class="upload-text" id="uploadFileName">Choose an image or drag & drop</div>
                <div class="upload-sub">Supports JPG, PNG, WEBP (Smartphone & Dashcam)</div>
                <input type="file" name="file" id="fileInput" required accept=".jpg,.jpeg,.png,.bmp,.webp">
            </div>
            <button type="submit" class="btn-primary" id="analyzeBtn">
                <i class="fa-solid fa-magnifying-glass"></i> Analyze Image
            </button>
        </form>
    </div>

    <!-- Live Camera Mode -->
    <div id="cameraSection" style="display: none;">
        <div class="camera-box" id="cameraBox">
            <video id="webcamVideo" autoplay playsinline muted></video>
            <canvas id="captureCanvas" style="display: none;"></canvas>
            <div class="live-badge">
                <div class="live-dot"></div>
                <span id="liveStatusText">LIVE FEED</span>
            </div>
        </div>
        <button class="btn-primary" id="btnCamToggle" onclick="toggleWebcam()">
            <i class="fa-solid fa-video"></i> <span id="camToggleText">Start Camera</span>
        </button>
    </div>

    <!-- Loading Spinner -->
    <div class="loading" id="loading">
        <div class="spinner"></div>
        <p style="font-size: 13px; color: #9ca3af;">Analyzing driver behavior...</p>
    </div>

    <!-- Error Box -->
    {% if error %}
    <div class="error-box">
        <i class="fa-solid fa-triangle-exclamation"></i> {{error}}
    </div>
    {% endif %}

    <!-- Preview Box -->
    <div id="clientPreviewBox" class="preview-box" style="display: {% if image %}block{% else %}none{% endif %};">
        <img id="clientPreviewImg" src="{% if image %}data:image/jpeg;base64,{{image}}{% endif %}" alt="Driver Frame">
    </div>

    <!-- Result Display -->
    <div class="result-card" id="resultCard" style="display: {% if label %}block{% else %}none{% endif %};">
        <div class="confidence-dial" id="confidenceDial" style="--conf-val: {{conf|default(0)}}; --dial-color: {% if label == 'c0' %}#00e676{% else %}#ff1744{% endif %};">
            <span id="confNum">{{conf|default(0)}}%</span>
        </div>
        <div class="result-label" id="resultDetail">{{detail}}</div>
        <div class="result-class" id="resultLabel">Predicted Class: {{label}}</div>

        <div class="top-predictions" id="topPredsContainer">
            <div class="top-predictions-title">Top Predictions:</div>
            <div id="topPredsList">
                {% if top_predictions %}
                {% for pred in top_predictions %}
                <div class="pred-item">
                    <span class="pred-name">{{pred.detail}}</span>
                    <span class="pred-conf">{{pred.confidence}}%</span>
                </div>
                {% endfor %}
                {% endif %}
            </div>
        </div>
    </div>

</div>

<script>
let webcamStream = null;
let liveInterval = null;

function setMode(mode) {
    const isUpload = mode === 'upload';
    document.getElementById('btnTabUpload').classList.toggle('active', isUpload);
    document.getElementById('btnTabCamera').classList.toggle('active', !isUpload);
    document.getElementById('uploadSection').style.display = isUpload ? 'block' : 'none';
    document.getElementById('cameraSection').style.display = !isUpload ? 'block' : 'none';

    if (isUpload && webcamStream) {
        stopWebcam();
    }
}

// File Input Change
document.getElementById('fileInput').addEventListener('change', function(e) {
    if (e.target.files.length > 0) {
        const file = e.target.files[0];
        document.getElementById('uploadFileName').textContent = file.name;
    }
});

// Upload Form Submit
document.getElementById('uploadForm').addEventListener('submit', function(e) {
    const fileInput = document.getElementById('fileInput');
    if (fileInput.files.length > 0) {
        document.getElementById('loading').style.display = 'block';
        const btn = document.getElementById('analyzeBtn');
        btn.disabled = true;
        btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Analyzing...';
    }
});

// Live Webcam Toggle
async function toggleWebcam() {
    if (!webcamStream) {
        try {
            webcamStream = await navigator.mediaDevices.getUserMedia({
                video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' }
            });
            const video = document.getElementById('webcamVideo');
            video.srcObject = webcamStream;
            document.getElementById('cameraBox').style.display = 'block';
            
            const btn = document.getElementById('btnCamToggle');
            btn.classList.add('btn-danger');
            document.getElementById('camToggleText').textContent = 'Stop Camera';

            // Real-time analysis every 800ms
            liveInterval = setInterval(analyzeWebcamFrame, 800);
        } catch (err) {
            alert('Unable to access camera: ' + err.message);
        }
    } else {
        stopWebcam();
    }
}

function stopWebcam() {
    if (webcamStream) {
        webcamStream.getTracks().forEach(t => t.stop());
        webcamStream = null;
    }
    clearInterval(liveInterval);
    document.getElementById('cameraBox').style.display = 'none';
    const btn = document.getElementById('btnCamToggle');
    btn.classList.remove('btn-danger');
    document.getElementById('camToggleText').textContent = 'Start Camera';
}

function analyzeWebcamFrame() {
    const video = document.getElementById('webcamVideo');
    const canvas = document.getElementById('captureCanvas');
    if (!video.videoWidth) return;

    canvas.width = 320;
    canvas.height = 240;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const base64Data = canvas.toDataURL('image/jpeg', 0.8).split(',')[1];

    fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: base64Data })
    })
    .then(r => r.json())
    .then(data => {
        if (data.success) {
            const pred = data.prediction;
            const card = document.getElementById('resultCard');
            card.style.display = 'block';

            const color = pred.is_safe ? '#00e676' : '#ff1744';
            const dial = document.getElementById('confidenceDial');
            dial.style.setProperty('--conf-val', pred.confidence);
            dial.style.setProperty('--dial-color', color);
            document.getElementById('confNum').textContent = pred.confidence + '%';
            document.getElementById('resultDetail').textContent = pred.detail;
            document.getElementById('resultLabel').textContent = 'Predicted Class: ' + pred.label;

            const list = document.getElementById('topPredsList');
            list.innerHTML = '';
            pred.top_predictions.forEach(p => {
                const item = document.createElement('div');
                item.className = 'pred-item';
                item.innerHTML = `
                    <span class="pred-name">${p.detail}</span>
                    <span class="pred-conf">${p.confidence}%</span>
                `;
                list.appendChild(item);
            });
        }
    })
    .catch(err => console.warn('Live frame error:', err));
}
</script>
</body>
</html>
"""

# ---------------- ROUTES ----------------
@app.route("/", methods=["GET", "POST"])
def index():
    label = detail = conf = image = error = None
    top_predictions = None

    if request.method == "POST":
        try:
            if "file" not in request.files or request.files["file"].filename == '':
                error = "Please select a valid image file"
                return render_template_string(HTML, error=error)

            file = request.files["file"]
            pil_img = Image.open(file)
            
            # Format preview base64
            buffered = io.BytesIO()
            pil_img.convert("RGB").save(buffered, format="JPEG", quality=90)
            image = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            # Run prediction
            result = predict_driver_state(pil_img)
            label = result['label']
            detail = result['detail']
            conf = result['confidence']
            top_predictions = result['top_predictions']

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            error = f"Error analyzing image: {str(e)}"

    return render_template_string(
        HTML, label=label, detail=detail, conf=conf,
        image=image, error=error, top_predictions=top_predictions
    )

@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        if request.is_json:
            b64_str = request.get_json().get("image")
            if not b64_str:
                return jsonify({"success": False, "error": "No image provided"}), 400
            if "," in b64_str:
                b64_str = b64_str.split(",")[1]
            img_bytes = base64.b64decode(b64_str)
        elif "file" in request.files:
            img_bytes = request.files["file"].read()
        else:
            return jsonify({"success": False, "error": "No valid image found"}), 400

        pil_img = Image.open(io.BytesIO(img_bytes))
        result = predict_driver_state(pil_img)
        return jsonify({"success": True, "prediction": result})

    except Exception as e:
        logger.error(f"API prediction error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model_loaded,
        "name": "DriverGuard AI",
        "timestamp": datetime.now().isoformat()
    })

# ---------------- RUN ----------------
if __name__ == "__main__":
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_ENV', 'development') == 'development'
    
    logger.info(f"Starting DriverGuard AI on {host}:{port}")
    app.run(debug=debug, host=host, port=port)
