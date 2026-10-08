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
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">

<style>
:root {
  --cyan:        #00f2fe;
  --cyan-dim:    rgba(0,242,254,0.12);
  --blue:        #4facfe;
  --safe:        #00e676;
  --danger:      #ff3d71;
  --warn:        #ffaa00;
  --surface-1:   #0d1117;
  --surface-2:   #161b27;
  --surface-3:   #1e2535;
  --surface-4:   #252d40;
  --border:      rgba(255,255,255,0.07);
  --text-1:      #f0f4f8;
  --text-2:      #8b9ab0;
  --text-3:      #4f5f73;
  --radius-xl:   20px;
  --radius-lg:   14px;
  --radius-md:   10px;
  --radius-sm:   6px;
  --transition:  all 0.2s cubic-bezier(0.4,0,0.2,1);
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html { scroll-behavior: smooth; }

body {
  font-family: 'Inter', -apple-system, sans-serif;
  background: var(--surface-1);
  color: var(--text-1);
  min-height: 100vh;
  overflow-x: hidden;
}

/* ── Animated background ── */
body::before {
  content: '';
  position: fixed;
  inset: 0;
  background:
    radial-gradient(ellipse 80% 50% at 10% 0%,   rgba(0,242,254,0.07) 0%, transparent 60%),
    radial-gradient(ellipse 60% 40% at 90% 100%,  rgba(79,172,254,0.07) 0%, transparent 60%),
    radial-gradient(ellipse 50% 30% at 50% 50%,   rgba(0,230,118,0.03) 0%, transparent 70%);
  pointer-events: none;
  z-index: 0;
}

/* ══════════════════════════════════════════
   HEADER
══════════════════════════════════════════ */
header {
  position: sticky;
  top: 0;
  z-index: 50;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 14px 36px;
  background: rgba(13,17,23,0.85);
  backdrop-filter: blur(20px);
  border-bottom: 1px solid var(--border);
}

.brand {
  display: flex;
  align-items: center;
  gap: 12px;
  text-decoration: none;
}

.brand-logo {
  width: 40px; height: 40px;
  border-radius: 12px;
  background: linear-gradient(135deg, var(--cyan), var(--blue));
  display: flex; align-items: center; justify-content: center;
  font-size: 20px; color: #0d1117;
  box-shadow: 0 0 20px rgba(0,242,254,0.35);
  flex-shrink: 0;
}

.brand-text .brand-name {
  font-size: 18px; font-weight: 800; letter-spacing: -0.4px;
  background: linear-gradient(120deg, #fff 0%, #c8d6e5 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  line-height: 1.1;
}

.brand-text .brand-sub {
  font-size: 10px; font-weight: 600; text-transform: uppercase;
  letter-spacing: 1.5px; color: var(--cyan);
  font-family: 'JetBrains Mono', monospace;
}

.status-chip {
  display: flex; align-items: center; gap: 8px;
  padding: 6px 14px;
  background: rgba(0,230,118,0.08);
  border: 1px solid rgba(0,230,118,0.2);
  border-radius: 30px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px; font-weight: 600; color: var(--safe);
}

.status-dot {
  width: 7px; height: 7px; border-radius: 50%;
  background: var(--safe); box-shadow: 0 0 8px var(--safe);
  animation: pulse-dot 2s ease infinite;
}

@keyframes pulse-dot {
  0%,100% { opacity:1; transform:scale(1); }
  50%      { opacity:0.4; transform:scale(0.75); }
}

/* ══════════════════════════════════════════
   MAIN LAYOUT  (two-column on wide screens)
══════════════════════════════════════════ */
.page-wrapper {
  position: relative; z-index: 1;
  max-width: 1200px;
  margin: 0 auto;
  padding: 40px 28px 60px;
  display: grid;
  grid-template-columns: 1fr 400px;
  gap: 28px;
  align-items: start;
}

@media (max-width: 860px) {
  .page-wrapper { grid-template-columns: 1fr; }
}

/* ══════════════════════════════════════════
   PANEL / CARD BASE
══════════════════════════════════════════ */
.panel {
  background: var(--surface-2);
  border: 1px solid var(--border);
  border-radius: var(--radius-xl);
  overflow: hidden;
  box-shadow: 0 20px 50px rgba(0,0,0,0.4);
  position: relative;
}

.panel::before {
  content: '';
  position: absolute; top: 0; left: 0; right: 0; height: 1px;
  background: linear-gradient(90deg, transparent, rgba(255,255,255,0.14), transparent);
}

.panel-header {
  padding: 20px 24px 0;
  display: flex; align-items: center; gap: 10px;
}

.panel-icon {
  width: 34px; height: 34px; border-radius: 9px;
  display: flex; align-items: center; justify-content: center;
  font-size: 15px; flex-shrink: 0;
}

.panel-icon.cyan  { background: var(--cyan-dim); color: var(--cyan); }
.panel-icon.green { background: rgba(0,230,118,0.12); color: var(--safe); }

.panel-title { font-size: 15px; font-weight: 700; color: var(--text-1); }
.panel-subtitle { font-size: 12px; color: var(--text-2); margin-top: 1px; }

.panel-body { padding: 20px 24px 24px; }

/* ══════════════════════════════════════════
   MODE SWITCHER (Upload / Camera)
══════════════════════════════════════════ */
.mode-switch {
  display: flex; gap: 6px;
  background: var(--surface-3);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 4px; margin-bottom: 20px;
}

.mode-switch button {
  flex: 1; background: transparent; border: none;
  padding: 9px 12px; border-radius: 10px;
  font-size: 13px; font-weight: 600; cursor: pointer;
  color: var(--text-2);
  display: flex; align-items: center; justify-content: center; gap: 7px;
  transition: var(--transition);
}

.mode-switch button.active {
  background: linear-gradient(135deg, var(--cyan), var(--blue));
  color: #0d1117;
  box-shadow: 0 4px 14px rgba(0,242,254,0.3);
}

/* ══════════════════════════════════════════
   DROPZONE
══════════════════════════════════════════ */
.dropzone {
  border: 2px dashed rgba(255,255,255,0.1);
  border-radius: var(--radius-lg);
  padding: 28px 16px;
  text-align: center;
  cursor: pointer;
  background: rgba(255,255,255,0.015);
  transition: var(--transition);
  margin-bottom: 14px;
}

.dropzone:hover, .dropzone.drag-over {
  border-color: var(--cyan);
  background: var(--cyan-dim);
  box-shadow: 0 0 24px rgba(0,242,254,0.1);
}

.dropzone-icon {
  font-size: 36px; margin-bottom: 10px;
  background: linear-gradient(135deg, var(--cyan), var(--blue));
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}

.dropzone-title { font-size: 14px; font-weight: 700; margin-bottom: 3px; }
.dropzone-hint  { font-size: 11px; color: var(--text-2); }

input[type=file] { display: none; }

/* ══════════════════════════════════════════
   BUTTONS
══════════════════════════════════════════ */
.btn {
  width: 100%; border: none; border-radius: var(--radius-lg);
  padding: 13px 18px; font-size: 14px; font-weight: 700;
  cursor: pointer; display: flex; align-items: center;
  justify-content: center; gap: 8px;
  transition: var(--transition);
}

.btn-primary {
  background: linear-gradient(135deg, var(--cyan), var(--blue));
  color: #0d1117;
  box-shadow: 0 4px 20px rgba(0,242,254,0.28);
}
.btn-primary:hover { transform: translateY(-2px); box-shadow: 0 8px 28px rgba(0,242,254,0.42); }
.btn-primary:disabled { opacity: 0.55; transform: none; }

.btn-red {
  background: linear-gradient(135deg, #ff3d71, #c80048);
  color: #fff;
  box-shadow: 0 4px 20px rgba(255,61,113,0.3);
}
.btn-red:hover { transform: translateY(-2px); box-shadow: 0 8px 28px rgba(255,61,113,0.45); }

/* ══════════════════════════════════════════
   IMAGE PREVIEW + CAMERA VIEWPORT
══════════════════════════════════════════ */
.viewport {
  border-radius: var(--radius-lg); overflow: hidden;
  background: #000; border: 1px solid var(--border);
  position: relative; margin-bottom: 14px;
  min-height: 220px; display: flex;
  align-items: center; justify-content: center;
}

.viewport img, .viewport video {
  width: 100%; height: auto;
  max-height: 360px; object-fit: contain; display: block;
}

.viewport video { transform: scaleX(-1); }

/* HUD corner brackets */
.hud {
  position: absolute; inset: 0;
  pointer-events: none;
  border: 1px solid rgba(0,242,254,0.15);
  border-radius: var(--radius-lg);
}
.hud-c { position:absolute; width:16px; height:16px; border-color:var(--cyan); border-style:solid; }
.hud-tl { top:8px;    left:8px;    border-width:2px 0 0 2px; }
.hud-tr { top:8px;    right:8px;   border-width:2px 2px 0 0; }
.hud-bl { bottom:8px; left:8px;    border-width:0 0 2px 2px; }
.hud-br { bottom:8px; right:8px;   border-width:0 2px 2px 0; }

.live-badge {
  position: absolute; top:10px; left:10px;
  display: flex; align-items:center; gap:6px;
  background: rgba(13,17,23,0.82); backdrop-filter: blur(8px);
  border: 1px solid var(--border);
  border-radius: 30px; padding: 4px 12px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px; font-weight: 600;
}

.live-rec { width:7px; height:7px; border-radius:50%; background:#ff3d71; box-shadow:0 0 8px #ff3d71; animation: pulse-dot 1s ease infinite; }

/* ══════════════════════════════════════════
   LOADING SPINNER
══════════════════════════════════════════ */
.loading-wrap {
  display: none; text-align: center; padding: 24px 0;
}

.spinner-ring {
  width: 40px; height: 40px; margin: 0 auto 12px;
  border: 3px solid rgba(0,242,254,0.15);
  border-top: 3px solid var(--cyan);
  border-radius: 50%;
  animation: spin 0.75s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

.loading-text { font-size: 13px; color: var(--text-2); font-weight: 500; }

/* ══════════════════════════════════════════
   ERROR
══════════════════════════════════════════ */
.error-box {
  background: rgba(255,61,113,0.1); border: 1px solid rgba(255,61,113,0.3);
  border-radius: var(--radius-md); padding: 12px 14px;
  font-size: 12px; color: #ff8fa8; margin-top: 14px;
  display: flex; align-items: flex-start; gap: 8px;
}

/* ══════════════════════════════════════════
   RIGHT COLUMN – RESULT PANEL
══════════════════════════════════════════ */
.result-panel-inner { padding: 24px; }

/* Confidence ring */
.ring-wrapper {
  display: flex; flex-direction: column; align-items: center;
  margin-bottom: 22px;
}

.ring-svg-container { position:relative; width:140px; height:140px; }

.ring-svg {
  width:140px; height:140px;
  transform: rotate(-90deg);
}

.ring-track { fill:none; stroke:rgba(255,255,255,0.06); stroke-width:10; }
.ring-fill  {
  fill:none; stroke-width:10; stroke-linecap:round;
  transition: stroke-dashoffset 0.8s cubic-bezier(0.4,0,0.2,1), stroke 0.4s ease;
}

.ring-center {
  position: absolute; inset: 0;
  display: flex; flex-direction: column;
  align-items: center; justify-content: center;
}

.ring-pct {
  font-family: 'JetBrains Mono', monospace;
  font-size: 28px; font-weight: 800; line-height: 1;
}

.ring-lbl { font-size: 10px; color: var(--text-2); text-transform: uppercase; letter-spacing: 0.8px; margin-top: 2px; }

/* Detection name tag */
.detection-name {
  margin-top: 14px; text-align: center;
}

.detection-badge {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 4px 14px; border-radius: 30px;
  font-size: 11px; font-weight: 700;
  text-transform: uppercase; letter-spacing: 0.8px;
  margin-bottom: 6px;
}

.detection-label { font-size: 18px; font-weight: 800; line-height: 1.2; }
.detection-class-tag {
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px; color: var(--text-2); margin-top: 3px;
}

/* Divider */
.divider {
  height: 1px; background: var(--border);
  margin: 20px 0;
}

/* Top predictions list */
.preds-title {
  font-size: 11px; font-weight: 700;
  text-transform: uppercase; letter-spacing: 1px;
  color: var(--text-3); margin-bottom: 12px;
}

.pred-row {
  margin-bottom: 12px;
}

.pred-meta {
  display: flex; justify-content: space-between;
  align-items: center; margin-bottom: 5px;
}

.pred-name { font-size: 13px; font-weight: 600; color: var(--text-1); display: flex; align-items:center; gap:6px; }
.pred-conf { font-family: 'JetBrains Mono', monospace; font-size: 13px; font-weight: 700; }

.pred-track {
  height: 6px; background: rgba(255,255,255,0.06);
  border-radius: 3px; overflow: hidden;
}

.pred-fill {
  height: 100%; border-radius: 3px;
  background: linear-gradient(90deg, var(--cyan), var(--blue));
  transition: width 0.7s cubic-bezier(0.4,0,0.2,1);
}

/* placeholder state */
.result-empty {
  display: flex; flex-direction: column;
  align-items: center; justify-content: center;
  padding: 40px 20px; text-align: center;
  color: var(--text-3);
}

.result-empty i { font-size: 48px; margin-bottom: 14px; opacity: 0.3; }
.result-empty p { font-size: 13px; line-height: 1.6; }

/* ══════════════════════════════════════════
   SMALL FILE INFO CHIP
══════════════════════════════════════════ */
.file-chip {
  display: inline-flex; align-items: center; gap: 6px;
  background: var(--surface-3); border: 1px solid var(--border);
  border-radius: var(--radius-sm); padding: 4px 10px;
  font-size: 11px; color: var(--text-2); margin-bottom: 12px;
}
</style>
</head>

<body>

<!-- ─── HEADER ─── -->
<header>
  <a class="brand" href="/">
    <div class="brand-logo"><i class="fa-solid fa-shield-halved"></i></div>
    <div class="brand-text">
      <div class="brand-name">DriverGuard AI</div>
      <div class="brand-sub">Driver Safety Intelligence</div>
    </div>
  </a>

</header>

<!-- ─── PAGE LAYOUT ─── -->
<div class="page-wrapper">

  <!-- ─── LEFT: Upload / Camera ─── -->
  <div class="panel">
    <div class="panel-header">
      <div class="panel-icon cyan"><i class="fa-solid fa-camera-viewfinder"></i></div>
      <div>
        <div class="panel-title">Driver Frame Analyzer</div>
        <div class="panel-subtitle">Upload a driver image or use your live camera feed</div>
      </div>
    </div>
    <div class="panel-body">

      <!-- Mode Switcher -->
      <div class="mode-switch">
        <button id="btnTabUpload" class="active" onclick="setMode('upload')">
          <i class="fa-solid fa-arrow-up-from-bracket"></i> Upload Image
        </button>
        <button id="btnTabCamera" onclick="setMode('camera')">
          <i class="fa-solid fa-video"></i> Live Camera
        </button>
      </div>

      <!-- UPLOAD SECTION -->
      <div id="uploadSection">
        <form method="post" enctype="multipart/form-data" id="uploadForm">
          <div class="dropzone" id="dropzone" onclick="document.getElementById('fileInput').click()">
            <div class="dropzone-icon"><i class="fa-solid fa-cloud-arrow-up"></i></div>
            <div class="dropzone-title" id="dropzoneTitle">Drop driver image here or click to browse</div>
            <div class="dropzone-hint">JPG · PNG · WEBP · JPEG &nbsp;•&nbsp; Dashcam / Smartphone photos</div>
            <input type="file" name="file" id="fileInput" accept=".jpg,.jpeg,.png,.bmp,.webp">
          </div>

          {% if image %}
          <div class="file-chip"><i class="fa-solid fa-file-image"></i> Image loaded</div>
          {% endif %}

          <div class="viewport" id="previewViewport" style="display:{% if image %}flex{% else %}none{% endif %}">
            <img id="previewImg" src="{% if image %}data:image/jpeg;base64,{{image}}{% endif %}" alt="Driver Frame">
            <div class="hud">
              <span class="hud-c hud-tl"></span><span class="hud-c hud-tr"></span>
              <span class="hud-c hud-bl"></span><span class="hud-c hud-br"></span>
            </div>
          </div>

          <button type="submit" class="btn btn-primary" id="analyzeBtn">
            <i class="fa-solid fa-magnifying-glass-chart"></i> Analyze Driver Behavior
          </button>
        </form>

        <!-- Spinner (shown while submitting) -->
        <div class="loading-wrap" id="loadingWrap">
          <div class="spinner-ring"></div>
          <div class="loading-text">Analyzing driver behavior…</div>
        </div>

        {% if error %}
        <div class="error-box">
          <i class="fa-solid fa-circle-exclamation" style="margin-top:1px"></i>
          <span>{{error}}</span>
        </div>
        {% endif %}
      </div>

      <!-- CAMERA SECTION -->
      <div id="cameraSection" style="display:none">
        <div class="viewport" id="cameraViewport" style="display:none; min-height:280px">
          <video id="webcamVideo" autoplay playsinline muted></video>
          <canvas id="captureCanvas" style="display:none"></canvas>
          <div class="live-badge">
            <span class="live-rec"></span> LIVE ANALYSIS
          </div>
          <div class="hud">
            <span class="hud-c hud-tl"></span><span class="hud-c hud-tr"></span>
            <span class="hud-c hud-bl"></span><span class="hud-c hud-br"></span>
          </div>
        </div>

        <button class="btn btn-primary" id="btnCamToggle" onclick="toggleWebcam()">
          <i class="fa-solid fa-video"></i> <span id="camBtnText">Start Camera</span>
        </button>
      </div>

    </div>
  </div>

  <!-- ─── RIGHT: Results ─── -->
  <div class="panel">
    <div class="panel-header">
      <div class="panel-icon green"><i class="fa-solid fa-gauge-high"></i></div>
      <div>
        <div class="panel-title">Detection Results</div>
        <div class="panel-subtitle">AI prediction with confidence breakdown</div>
      </div>
    </div>

    <!-- No result yet -->
    <div id="resultEmpty" class="result-empty" style="display:{% if label %}none{% else %}flex{% endif %}">
      <i class="fa-regular fa-eye-slash"></i>
      <p>Upload a driver photo or start the live camera to see real-time distraction analysis.</p>
    </div>

    <!-- Result content -->
    <div class="result-panel-inner" id="resultContent" style="display:{% if label %}block{% else %}none{% endif %}">

      <!-- Confidence Ring -->
      <div class="ring-wrapper">
        <div class="ring-svg-container">
          <svg class="ring-svg" viewBox="0 0 140 140">
            <circle class="ring-track" cx="70" cy="70" r="60"></circle>
            <circle class="ring-fill" id="ringFill" cx="70" cy="70" r="60"
              stroke="{{ '#00e676' if label == 'c0' else '#ff3d71' if label else '#334155' }}"
              stroke-dasharray="376.99"
              stroke-dashoffset="{{ 376.99 - ((conf or 0) / 100 * 376.99) }}">
            </circle>
          </svg>
          <div class="ring-center">
            <span class="ring-pct" id="ringPct" style="color:{{ '#00e676' if label == 'c0' else '#ff3d71' if label else '#334155' }}">{{ conf or 0 }}%</span>
            <span class="ring-lbl">Confidence</span>
          </div>
        </div>

        <div class="detection-name">
          {% if label %}
          <div class="detection-badge" id="detBadge"
            style="background:{{ 'rgba(0,230,118,0.12)' if label == 'c0' else 'rgba(255,61,113,0.12)' }};
                   color:{{ '#00e676' if label == 'c0' else '#ff3d71' }}">
            <i class="fa-solid {{ 'fa-shield-check' if label == 'c0' else 'fa-triangle-exclamation' }}"></i>
            {{ 'Safe' if label == 'c0' else 'Distraction Detected' }}
          </div>
          {% endif %}
          <div class="detection-label" id="detLabel">{{detail|default('')}}</div>
          <div class="detection-class-tag" id="detClass">{% if label %}Class: {{label}}{% endif %}</div>
        </div>
      </div>

      <div class="divider"></div>

      <!-- Top Predictions Breakdown -->
      <div class="preds-title">Top Predictions</div>
      <div id="predsContainer">
        {% if top_predictions %}
        {% for pred in top_predictions %}
        <div class="pred-row">
          <div class="pred-meta">
            <span class="pred-name">
              <i class="fa-solid fa-circle-dot" style="color: {% if pred.label == 'c0' %}#00e676{% else %}rgba(255,255,255,0.2){% endif %}; font-size:9px"></i>
              {{pred.detail}}
            </span>
            <span class="pred-conf" style="color:{% if loop.first %}var(--cyan){% else %}var(--text-2){% endif %}">{{pred.confidence}}%</span>
          </div>
          <div class="pred-track">
            <div class="pred-fill" style="width:{{pred.confidence}}%; {% if not loop.first %}background: rgba(255,255,255,0.15);{% endif %}"></div>
          </div>
        </div>
        {% endfor %}
        {% endif %}
      </div>

    </div>
  </div>

</div><!-- /page-wrapper -->

<script>
let webcamStream = null;
let liveInterval  = null;

/* ── Mode Switch ── */
function setMode(mode) {
  const toUpload = mode === 'upload';
  document.getElementById('btnTabUpload').classList.toggle('active', toUpload);
  document.getElementById('btnTabCamera').classList.toggle('active', !toUpload);
  document.getElementById('uploadSection').style.display  = toUpload ? 'block' : 'none';
  document.getElementById('cameraSection').style.display  = toUpload ? 'none'  : 'block';
  if (toUpload && webcamStream) stopWebcam();
}

/* ── File Drag & Drop ── */
const dz = document.getElementById('dropzone');
['dragenter','dragover'].forEach(e => dz.addEventListener(e, ev => { ev.preventDefault(); dz.classList.add('drag-over'); }));
['dragleave','drop'].forEach(e => dz.addEventListener(e, ev => { ev.preventDefault(); dz.classList.remove('drag-over'); }));
dz.addEventListener('drop', ev => {
  if (ev.dataTransfer.files.length) { handleFile(ev.dataTransfer.files[0]); }
});

/* ── File Input ── */
document.getElementById('fileInput').addEventListener('change', function(e) {
  if (e.target.files.length) handleFile(e.target.files[0]);
});

function handleFile(file) {
  document.getElementById('dropzoneTitle').textContent = file.name;
  const reader = new FileReader();
  reader.onload = e => {
    const preview = document.getElementById('previewViewport');
    const img     = document.getElementById('previewImg');
    img.src = e.target.result;
    preview.style.display = 'flex';
  };
  reader.readAsDataURL(file);
}

/* ── Form Submit Spinner ── */
document.getElementById('uploadForm').addEventListener('submit', function() {
  const fi = document.getElementById('fileInput');
  if (fi.files.length) {
    document.getElementById('loadingWrap').style.display = 'block';
    const btn = document.getElementById('analyzeBtn');
    btn.disabled = true;
    btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Analyzing…';
  }
});

/* ── Clipboard Paste ── */
window.addEventListener('paste', e => {
  const items = (e.clipboardData || e.originalEvent.clipboardData).items;
  for (const item of items) {
    if (item.type.startsWith('image')) { handleFile(item.getAsFile()); break; }
  }
});

/* ── Webcam ── */
async function toggleWebcam() {
  if (!webcamStream) {
    try {
      webcamStream = await navigator.mediaDevices.getUserMedia({
        video: { width:{ideal:640}, height:{ideal:480}, facingMode:'user' }
      });
      const video = document.getElementById('webcamVideo');
      video.srcObject = webcamStream;
      document.getElementById('cameraViewport').style.display = 'flex';
      document.getElementById('btnCamToggle').classList.replace('btn-primary','btn-red');
      document.getElementById('camBtnText').textContent = 'Stop Camera';
      liveInterval = setInterval(captureAndAnalyze, 900);
    } catch(err) {
      alert('Camera access denied: ' + err.message);
    }
  } else {
    stopWebcam();
  }
}

function stopWebcam() {
  webcamStream?.getTracks().forEach(t => t.stop());
  webcamStream = null;
  clearInterval(liveInterval);
  document.getElementById('cameraViewport').style.display  = 'none';
  document.getElementById('btnCamToggle').classList.replace('btn-red','btn-primary');
  document.getElementById('camBtnText').textContent = 'Start Camera';
}

function captureAndAnalyze() {
  const video  = document.getElementById('webcamVideo');
  const canvas = document.getElementById('captureCanvas');
  if (!video.videoWidth) return;
  canvas.width = 320; canvas.height = 240;
  canvas.getContext('2d').drawImage(video, 0, 0, 320, 240);
  const b64 = canvas.toDataURL('image/jpeg', 0.82).split(',')[1];

  fetch('/api/predict', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({ image: b64 })
  })
  .then(r => r.json())
  .then(data => { if (data.success) renderResult(data.prediction); })
  .catch(e => console.warn('Frame error:', e));
}

/* ── Render Results (for live camera) ── */
function renderResult(pred) {
  document.getElementById('resultEmpty').style.display   = 'none';
  document.getElementById('resultContent').style.display = 'block';

  const isSafe  = pred.is_safe;
  const color   = isSafe ? '#00e676' : '#ff3d71';
  const circumf = 376.99;
  const offset  = circumf - (pred.confidence / 100 * circumf);

  /* ring */
  const ring = document.getElementById('ringFill');
  ring.style.stroke          = color;
  ring.style.strokeDashoffset = offset;

  const pct = document.getElementById('ringPct');
  pct.textContent  = pred.confidence + '%';
  pct.style.color  = color;

  /* badge */
  const badge = document.getElementById('detBadge');
  badge.style.background = isSafe ? 'rgba(0,230,118,0.12)' : 'rgba(255,61,113,0.12)';
  badge.style.color      = color;
  badge.innerHTML = `<i class="fa-solid ${isSafe ? 'fa-shield-check' : 'fa-triangle-exclamation'}"></i> ${isSafe ? 'Safe' : 'Distraction Detected'}`;

  document.getElementById('detLabel').textContent = pred.detail;
  document.getElementById('detClass').textContent = 'Class: ' + pred.label;

  /* predictions */
  const container = document.getElementById('predsContainer');
  container.innerHTML = '';
  pred.top_predictions.forEach((p, i) => {
    const accent = i === 0 ? 'var(--cyan)' : 'var(--text-2)';
    const barBg  = i === 0 ? 'linear-gradient(90deg,var(--cyan),var(--blue))' : 'rgba(255,255,255,0.15)';
    container.innerHTML += `
      <div class="pred-row">
        <div class="pred-meta">
          <span class="pred-name">
            <i class="fa-solid fa-circle-dot" style="color:${p.label==='c0'?'#00e676':'rgba(255,255,255,0.2)'};font-size:9px"></i>
            ${p.detail}
          </span>
          <span class="pred-conf" style="color:${accent}">${p.confidence}%</span>
        </div>
        <div class="pred-track"><div class="pred-fill" style="width:${p.confidence}%;background:${barBg}"></div></div>
      </div>`;
  });
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
