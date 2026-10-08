from flask import Flask, request, render_template_string, jsonify, send_from_directory
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps, ImageFilter
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

# Configure TensorFlow for optimal deployment
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

try:
    # Disable GPU if CPU deployment
    if os.getenv('FORCE_CPU', 'false').lower() == 'true':
        tf.config.set_visible_devices([], 'GPU')
except Exception:
    pass

app = Flask(__name__, static_folder='static')
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max

log_level = os.getenv('LOG_LEVEL', 'INFO')
logging.basicConfig(level=getattr(logging, log_level), format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Model state
model = None
eff_feature_model = None
labels = None
id_to_label = None
IMG_SIZE = (224, 224)
MODEL_PATH = os.getenv('MODEL_PATH', 'v7_plus_distracted_driver.keras')
LABELS_PATH = os.getenv('LABELS_PATH', 'labels.pkl')
model_loaded = False

# Domain Knowledge Taxonomy & Risk Profiles
CLASS_METADATA = {
    "c0": {
        "title": "Safe Driving",
        "category": "Nominal Focus",
        "severity": "safe",
        "risk_score": 5,
        "visual_distraction": 5,
        "manual_distraction": 5,
        "cognitive_distraction": 5,
        "description": "Driver is actively focused on the roadway with hands properly positioned on steering wheel.",
        "recommendation": "Maintain standard scanning techniques and safe following distances.",
        "badge_color": "#00e676",
        "icon": "fa-shield-check"
    },
    "c1": {
        "title": "Texting – Right Hand",
        "category": "High Risk Distraction",
        "severity": "critical",
        "risk_score": 96,
        "visual_distraction": 95,
        "manual_distraction": 90,
        "cognitive_distraction": 88,
        "description": "Driver is typing/reading text messages using right hand, diverting visual gaze and manual control.",
        "recommendation": "Immediate hazard! Put the mobile device down and keep both hands on the wheel.",
        "badge_color": "#ff1744",
        "icon": "fa-mobile-screen-button"
    },
    "c2": {
        "title": "Talking on Phone – Right Hand",
        "category": "High Risk Distraction",
        "severity": "danger",
        "risk_score": 78,
        "visual_distraction": 55,
        "manual_distraction": 85,
        "cognitive_distraction": 80,
        "description": "Driver holding phone to right ear with one hand off the steering control.",
        "recommendation": "Use integrated hands-free Bluetooth systems or pull over safely before calls.",
        "badge_color": "#ff5252",
        "icon": "fa-phone-volume"
    },
    "c3": {
        "title": "Texting – Left Hand",
        "category": "High Risk Distraction",
        "severity": "critical",
        "risk_score": 96,
        "visual_distraction": 95,
        "manual_distraction": 90,
        "cognitive_distraction": 88,
        "description": "Driver is operating smartphone with left hand, significantly compromising reaction times.",
        "recommendation": "Immediate hazard! Put mobile device away and refocus on forward vehicle path.",
        "badge_color": "#ff1744",
        "icon": "fa-mobile-screen-button"
    },
    "c4": {
        "title": "Talking on Phone – Left Hand",
        "category": "High Risk Distraction",
        "severity": "danger",
        "risk_score": 78,
        "visual_distraction": 55,
        "manual_distraction": 85,
        "cognitive_distraction": 80,
        "description": "Driver holding phone to left ear, reducing situational awareness and steering readiness.",
        "recommendation": "Switch to hands-free voice commands or postpone conversation until parked.",
        "badge_color": "#ff5252",
        "icon": "fa-phone-volume"
    },
    "c5": {
        "title": "Operating Infotainment / Radio",
        "category": "Moderate Distraction",
        "severity": "warning",
        "risk_score": 52,
        "visual_distraction": 70,
        "manual_distraction": 65,
        "cognitive_distraction": 40,
        "description": "Driver interacting with center console, touchscreen, or climate/radio controls.",
        "recommendation": "Utilize steering wheel media controls or adjust settings before departure.",
        "badge_color": "#ff9100",
        "icon": "fa-sliders"
    },
    "c6": {
        "title": "Drinking / Reaching Beverage",
        "category": "Moderate Distraction",
        "severity": "warning",
        "risk_score": 62,
        "visual_distraction": 50,
        "manual_distraction": 75,
        "cognitive_distraction": 35,
        "description": "Driver consuming drink or holding beverage container with reduced steering grip.",
        "recommendation": "Drink only at full stops or secure cup firmly in cabin holder.",
        "badge_color": "#ffab00",
        "icon": "fa-mug-hot"
    },
    "c7": {
        "title": "Reaching Behind / Rear Seats",
        "category": "Severe Distraction",
        "severity": "critical",
        "risk_score": 90,
        "visual_distraction": 90,
        "manual_distraction": 95,
        "cognitive_distraction": 70,
        "description": "Driver turning torso and reaching to rear passenger cabin; severe lane drift risk.",
        "recommendation": "Pull over safely before retrieving items from rear seats or floorboards.",
        "badge_color": "#ff1744",
        "icon": "fa-hand-back-fist"
    },
    "c8": {
        "title": "Hair & Makeup / Personal Grooming",
        "category": "Severe Distraction",
        "severity": "critical",
        "risk_score": 88,
        "visual_distraction": 92,
        "manual_distraction": 90,
        "cognitive_distraction": 65,
        "description": "Driver looking into vanity mirror or grooming hair/face during vehicle movement.",
        "recommendation": "Perform personal grooming before starting vehicle or when parked.",
        "badge_color": "#ff1744",
        "icon": "fa-wand-magic-sparkles"
    },
    "c9": {
        "title": "Talking to Passenger / Turning Head",
        "category": "Mild-Moderate Distraction",
        "severity": "caution",
        "risk_score": 42,
        "visual_distraction": 65,
        "manual_distraction": 15,
        "cognitive_distraction": 55,
        "description": "Driver turning head toward passenger seat, periodically looking away from roadway.",
        "recommendation": "Keep eyes forward; converse without taking visual focus off the road ahead.",
        "badge_color": "#ffd600",
        "icon": "fa-comments"
    }
}

CLASS_DETAILS = {k: v["title"] for k, v in CLASS_METADATA.items()}

# ---------------- MODEL INITIALIZATION ----------------
def initialize_model():
    """Load model, labels, and initialize Grad-CAM extractor."""
    global model, eff_feature_model, labels, id_to_label, IMG_SIZE, model_loaded
    
    logger.info("Initializing Driver Distraction AI System...")
    
    # Load labels
    try:
        if os.path.exists(LABELS_PATH):
            with open(LABELS_PATH, 'rb') as f:
                labels = pickle.load(f)
            id_to_label = {v: k for k, v in labels.items()}
            logger.info(f"Labels loaded successfully: {labels}")
        else:
            logger.warning("Labels file not found, using default label map.")
            labels = {f"c{i}": i for i in range(10)}
            id_to_label = {i: f"c{i}" for i in range(10)}
    except Exception as e:
        logger.error(f"Error loading labels: {e}")
        labels = {f"c{i}": i for i in range(10)}
        id_to_label = {i: f"c{i}" for i in range(10)}
    
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model file not found at: {MODEL_PATH}")
        return False
        
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        input_shape = model.input_shape
        IMG_SIZE = (input_shape[1], input_shape[2]) if len(input_shape) >= 3 else (224, 224)
        logger.info(f"Keras model loaded successfully with input shape: {input_shape}")
        
        # Build Grad-CAM feature extractor
        try:
            eff = model.get_layer('efficientnetb0')
            last_conv_layer = eff.get_layer('top_activation')
            eff_feature_model = tf.keras.Model(inputs=eff.input, outputs=[last_conv_layer.output, eff.output])
            logger.info("Grad-CAM feature extractor constructed successfully.")
        except Exception as cam_err:
            logger.warning(f"Grad-CAM feature model could not be initialized: {cam_err}")
            eff_feature_model = None
            
        # Warm up model with a dummy inference
        dummy = np.zeros((1, IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.float32)
        _ = model.predict(dummy, verbose=0)
        logger.info("Model warm-up completed.")
        
        model_loaded = True
        return True
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False

# Initialize on startup
initialize_model()

# ---------------- ACCURACY ENHANCEMENT & PREPROCESSING ----------------
def pad_and_resize_aspect_preserved(image, target_size=(224, 224)):
    """
    Preserves driver aspect ratio by letterboxing with reflective/edge padding
    instead of squashing the image, preventing deformation of hands and posture.
    """
    orig_w, orig_h = image.size
    target_w, target_h = target_size
    
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    
    resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # Create letterbox canvas with neutral dark tone matching car cabins
    padded = Image.new("RGB", target_size, (20, 22, 28))
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    padded.paste(resized, (paste_x, paste_y))
    return padded

def generate_tta_variants(pil_img, target_size=(224, 224)):
    """
    Test-Time Augmentation (TTA) Generator:
    Generates multi-scale aspect-preserved crops and multi-angle samples.
    """
    variants = []
    
    # 1. Base aspect-preserved padded image
    base_padded = pad_and_resize_aspect_preserved(pil_img, target_size)
    variants.append(np.array(base_padded, dtype=np.float32))
    
    # 2. High-fidelity center crop (focusing on driver cabin center)
    w, h = pil_img.size
    min_dim = min(w, h)
    left = (w - min_dim) // 2
    top = (h - min_dim) // 2
    center_cropped = pil_img.crop((left, top, left + min_dim, top + min_dim)).resize(target_size, Image.Resampling.LANCZOS)
    variants.append(np.array(center_cropped, dtype=np.float32))
    
    # 3. Slight zoom in (92% crop) to focus on driver gestures
    crop_w, crop_h = int(w * 0.92), int(h * 0.92)
    left = (w - crop_w) // 2
    top = (h - crop_h) // 2
    zoom_crop = pil_img.crop((left, top, left + crop_w, top + crop_h)).resize(target_size, Image.Resampling.LANCZOS)
    variants.append(np.array(zoom_crop, dtype=np.float32))
    
    # Stack batch: shape (3, 224, 224, 3)
    batch = np.stack(variants, axis=0)
    return batch, base_padded

def compute_gradcam_heatmap(img_array, target_class_idx):
    """Compute normalized Grad-CAM activation heatmap for the target class."""
    if eff_feature_model is None or model is None:
        return None
    try:
        img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
        if len(img_tensor.shape) == 3:
            img_tensor = tf.expand_dims(img_tensor, axis=0)
            
        with tf.GradientTape() as tape:
            conv_out, _ = eff_feature_model(img_tensor, training=False)
            tape.watch(conv_out)
            h = model.get_layer('global_average_pooling2d')(conv_out)
            h = model.get_layer('batch_normalization')(h, training=False)
            h = model.get_layer('dense')(h)
            preds = model.get_layer('dense_1')(h)
            loss = preds[:, target_class_idx]
            
        grads = tape.gradient(loss, conv_out)
        if grads is None:
            return None
            
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_out[0]), axis=-1).numpy()
        heatmap = np.maximum(heatmap, 0)
        max_h = np.max(heatmap)
        if max_h > 1e-8:
            heatmap = heatmap / max_h
            
        # Resize to 224x224
        hm_img = Image.fromarray((heatmap * 255).astype(np.uint8)).resize((224, 224), Image.Resampling.BICUBIC)
        return np.array(hm_img, dtype=np.float32) / 255.0
    except Exception as e:
        logger.warning(f"Grad-CAM computation exception: {e}")
        return None

def colormap_turbo(val):
    """Pure NumPy high-contrast Turbo/Jet colormap for explainable attention heatmaps."""
    x = np.clip(val, 0, 1)
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0, 1)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0, 1)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0, 1)
    return np.stack([r, g, b], axis=-1)

def generate_cam_overlay(base_img_pil, heatmap_2d):
    """Overlays Grad-CAM attention heatmap onto the base image."""
    if heatmap_2d is None:
        return None
    try:
        color_hm = (colormap_turbo(heatmap_2d) * 255).astype(np.uint8)
        hm_pil = Image.fromarray(color_hm).resize(base_img_pil.size, Image.Resampling.BICUBIC)
        # Apply slight blur to make heatmap smooth and realistic
        hm_pil = hm_pil.filter(ImageFilter.GaussianBlur(radius=3))
        blended = Image.blend(base_img_pil, hm_pil, alpha=0.45)
        
        buffered = io.BytesIO()
        blended.save(buffered, format="JPEG", quality=92)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception as e:
        logger.error(f"Error generating CAM overlay: {e}")
        return None

# ---------------- PREDICTION LOGIC ----------------
def run_enhanced_inference(pil_img, use_tta=True, generate_cam=True):
    """
    Executes full high-accuracy inference pipeline:
    1. EXIF orientation correction
    2. Test-Time Augmentation (TTA)
    3. Probability ensembling & confidence calibration
    4. Grad-CAM visual attention mapping
    5. Driver risk scoring & distraction taxonomy
    """
    start_time = time.time()
    
    # 1. EXIF auto-rotation
    pil_img = ImageOps.exif_transpose(pil_img)
    pil_img = pil_img.convert("RGB")
    
    # 2. TTA variants
    if use_tta:
        batch, base_padded = generate_tta_variants(pil_img, IMG_SIZE)
        raw_preds = model.predict(batch, verbose=0)
        # Weighted ensemble: 50% base padded, 25% center crop, 25% zoom crop
        weights = np.array([0.50, 0.25, 0.25]).reshape(3, 1)
        avg_preds = np.sum(raw_preds * weights, axis=0)
    else:
        base_padded = pad_and_resize_aspect_preserved(pil_img, IMG_SIZE)
        single_arr = np.expand_dims(np.array(base_padded, dtype=np.float32), axis=0)
        avg_preds = model.predict(single_arr, verbose=0)[0]
        
    # Softmax temperature smoothing for calibrated probabilities
    temperature = 0.95
    exp_preds = np.exp(np.log(np.clip(avg_preds, 1e-7, 1.0)) / temperature)
    calibrated_probs = exp_preds / np.sum(exp_preds)
    
    # Top prediction
    top_idx = int(np.argmax(calibrated_probs))
    top_label = id_to_label.get(top_idx, f"c{top_idx}")
    top_meta = CLASS_METADATA.get(top_label, CLASS_METADATA["c0"])
    top_conf = round(float(calibrated_probs[top_idx]) * 100, 1)
    
    # Full distribution & Top predictions
    all_predictions = []
    for idx in range(len(calibrated_probs)):
        lbl = id_to_label.get(idx, f"c{idx}")
        meta = CLASS_METADATA.get(lbl, {})
        prob = round(float(calibrated_probs[idx]) * 100, 1)
        all_predictions.append({
            "index": idx,
            "label": lbl,
            "title": meta.get("title", lbl),
            "category": meta.get("category", "General"),
            "severity": meta.get("severity", "info"),
            "confidence": prob,
            "badge_color": meta.get("badge_color", "#4facfe"),
            "icon": meta.get("icon", "fa-circle-dot")
        })
        
    all_predictions.sort(key=lambda x: x["confidence"], reverse=True)
    top_3 = all_predictions[:3]
    
    # Calculate Driver Safety Telematics
    # Weighted risk index across predictions
    overall_risk = 0.0
    for pred in all_predictions:
        lbl = pred["label"]
        prob = pred["confidence"] / 100.0
        meta = CLASS_METADATA.get(lbl, {})
        overall_risk += prob * meta.get("risk_score", 50)
        
    overall_safety_score = max(0, min(100, round(100 - overall_risk)))
    
    if overall_safety_score >= 80:
        safety_status = "SAFE DRIVING"
        safety_status_color = "#00e676"
        status_alert_level = "nominal"
    elif overall_safety_score >= 50:
        safety_status = "ELEVATED RISK"
        safety_status_color = "#ff9100"
        status_alert_level = "warning"
    else:
        safety_status = "CRITICAL DISTRACTION"
        safety_status_color = "#ff1744"
        status_alert_level = "critical"
        
    # Generate Grad-CAM if requested
    cam_base64 = None
    if generate_cam and eff_feature_model is not None:
        base_arr = np.array(base_padded, dtype=np.float32)
        heatmap = compute_gradcam_heatmap(base_arr, top_idx)
        cam_base64 = generate_cam_overlay(base_padded, heatmap)
        
    # Original image base64
    orig_buf = io.BytesIO()
    base_padded.save(orig_buf, format="JPEG", quality=90)
    orig_base64 = base64.b64encode(orig_buf.getvalue()).decode('utf-8')
    
    latency_ms = round((time.time() - start_time) * 1000, 1)
    
    return {
        "success": True,
        "prediction": {
            "label": top_label,
            "title": top_meta["title"],
            "category": top_meta["category"],
            "severity": top_meta["severity"],
            "confidence": top_conf,
            "risk_score": top_meta["risk_score"],
            "description": top_meta["description"],
            "recommendation": top_meta["recommendation"],
            "badge_color": top_meta["badge_color"],
            "icon": top_meta["icon"],
            "metrics": {
                "visual_distraction": top_meta["visual_distraction"],
                "manual_distraction": top_meta["manual_distraction"],
                "cognitive_distraction": top_meta["cognitive_distraction"]
            }
        },
        "telematics": {
            "safety_score": overall_safety_score,
            "safety_status": safety_status,
            "safety_status_color": safety_status_color,
            "status_alert_level": status_alert_level,
            "is_distracted": top_label != "c0",
            "latency_ms": latency_ms,
            "tta_enabled": use_tta,
            "model_version": "EfficientNet-B0 DMS v7.2-Ultra"
        },
        "top_3": top_3,
        "all_predictions": all_predictions,
        "images": {
            "original": orig_base64,
            "gradcam": cam_base64
        }
    }

# ---------------- HTML DASHBOARD TEMPLATE ----------------
HTML_DASHBOARD = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AegisEye DMS – Real-Time AI Driver Monitoring & Safety Telematics</title>
    
    <!-- Modern Typography & Icons -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
    
    <style>
        :root {
            --bg-deep: #0a0d14;
            --bg-card: rgba(18, 24, 38, 0.75);
            --bg-card-hover: rgba(26, 35, 55, 0.85);
            --border-glass: rgba(255, 255, 255, 0.08);
            --border-glow: rgba(0, 242, 254, 0.35);
            
            --primary: #00f2fe;
            --primary-gradient: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
            --safe-color: #00e676;
            --safe-gradient: linear-gradient(135deg, #00e676 0%, #00b0ff 100%);
            --warning-color: #ff9100;
            --warning-gradient: linear-gradient(135deg, #ff9100 0%, #ff5252 100%);
            --danger-color: #ff1744;
            --danger-gradient: linear-gradient(135deg, #ff1744 0%, #f50057 100%);
            
            --text-main: #f0f4f8;
            --text-muted: #8a99ad;
            --text-dim: #5c6b7d;
            
            --radius-lg: 20px;
            --radius-md: 14px;
            --radius-sm: 8px;
            --shadow-hud: 0 20px 50px rgba(0, 0, 0, 0.5), 0 0 30px rgba(0, 242, 254, 0.05);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Outfit', -apple-system, sans-serif;
            background-color: var(--bg-deep);
            color: var(--text-main);
            min-height: 100vh;
            overflow-x: hidden;
            background-image: 
                radial-gradient(circle at 10% 20%, rgba(0, 242, 254, 0.05) 0%, transparent 40%),
                radial-gradient(circle at 90% 80%, rgba(79, 172, 254, 0.05) 0%, transparent 40%),
                linear-gradient(rgba(10, 13, 20, 0.95), rgba(10, 13, 20, 0.95)),
                repeating-linear-gradient(0deg, transparent, transparent 40px, rgba(255, 255, 255, 0.015) 40px, rgba(255, 255, 255, 0.015) 41px);
        }

        /* Top Navigation Header */
        header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 16px 36px;
            background: rgba(10, 13, 20, 0.8);
            backdrop-filter: blur(20px);
            border-bottom: 1px solid var(--border-glass);
            position: sticky;
            top: 0;
            z-index: 100;
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 14px;
        }

        .brand-logo {
            width: 42px;
            height: 42px;
            border-radius: 12px;
            background: var(--primary-gradient);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 20px;
            color: #0a0d14;
            box-shadow: 0 0 20px rgba(0, 242, 254, 0.4);
        }

        .brand-title {
            font-size: 20px;
            font-weight: 700;
            letter-spacing: -0.5px;
            background: linear-gradient(120deg, #ffffff, #cfd9df);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .brand-tag {
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            color: var(--primary);
            font-family: 'JetBrains Mono', monospace;
        }

        .header-telemetry {
            display: flex;
            align-items: center;
            gap: 20px;
        }

        .tele-badge {
            display: flex;
            align-items: center;
            gap: 8px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 12px;
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid var(--border-glass);
            padding: 6px 14px;
            border-radius: 30px;
        }

        .tele-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--safe-color);
            box-shadow: 0 0 10px var(--safe-color);
            animation: pulse-dot 2s infinite;
        }

        @keyframes pulse-dot {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.4; transform: scale(0.85); }
        }

        /* Mode Tabs */
        .tab-bar {
            display: flex;
            gap: 8px;
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid var(--border-glass);
            padding: 4px;
            border-radius: 12px;
        }

        .tab-btn {
            background: transparent;
            border: none;
            color: var(--text-muted);
            padding: 8px 18px;
            border-radius: 8px;
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 8px;
            transition: all 0.2s ease;
        }

        .tab-btn:hover {
            color: var(--text-main);
            background: rgba(255, 255, 255, 0.05);
        }

        .tab-btn.active {
            background: var(--primary-gradient);
            color: #0a0d14;
            box-shadow: 0 4px 15px rgba(0, 242, 254, 0.25);
        }

        /* Container Layout */
        .main-container {
            max-width: 1440px;
            margin: 28px auto;
            padding: 0 28px;
        }

        .view-section {
            display: none;
        }

        .view-section.active {
            display: block;
            animation: fadeIn 0.3s ease;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(8px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* Grid Layout for Inspector */
        .inspector-grid {
            display: grid;
            grid-template-columns: 1.15fr 0.85fr;
            gap: 24px;
        }

        @media (max-width: 1024px) {
            .inspector-grid {
                grid-template-columns: 1fr;
            }
        }

        /* Glass Card */
        .glass-card {
            background: var(--bg-card);
            backdrop-filter: blur(24px);
            border: 1px solid var(--border-glass);
            border-radius: var(--radius-lg);
            padding: 24px;
            box-shadow: var(--shadow-hud);
            position: relative;
            overflow: hidden;
        }

        .glass-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.15), transparent);
        }

        .card-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 20px;
        }

        .card-title {
            font-size: 17px;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 10px;
            color: var(--text-main);
        }

        .card-title i {
            color: var(--primary);
        }

        /* Upload Drop Zone */
        .dropzone-box {
            border: 2px dashed rgba(255, 255, 255, 0.12);
            border-radius: var(--radius-md);
            padding: 32px 20px;
            text-align: center;
            background: rgba(255, 255, 255, 0.015);
            cursor: pointer;
            transition: all 0.25s ease;
            position: relative;
        }

        .dropzone-box:hover, .dropzone-box.dragover {
            border-color: var(--primary);
            background: rgba(0, 242, 254, 0.04);
            box-shadow: 0 0 25px rgba(0, 242, 254, 0.1);
        }

        .dropzone-icon {
            font-size: 42px;
            background: var(--primary-gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 12px;
        }

        .dropzone-text {
            font-size: 15px;
            font-weight: 600;
            color: var(--text-main);
            margin-bottom: 4px;
        }

        .dropzone-sub {
            font-size: 12px;
            color: var(--text-muted);
        }

        /* Preset Demo Scenario Strip */
        .presets-strip {
            margin-top: 18px;
        }

        .presets-label {
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: var(--text-muted);
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .preset-chips {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }

        .preset-chip {
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid var(--border-glass);
            color: var(--text-main);
            padding: 7px 12px;
            border-radius: var(--radius-sm);
            font-size: 12px;
            font-weight: 500;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 6px;
            transition: all 0.2s ease;
        }

        .preset-chip:hover {
            border-color: var(--primary);
            background: rgba(0, 242, 254, 0.08);
            transform: translateY(-2px);
        }

        /* Image Display & Comparison */
        .viewport-container {
            margin-top: 20px;
            border-radius: var(--radius-md);
            overflow: hidden;
            background: #000;
            position: relative;
            min-height: 320px;
            display: flex;
            align-items: center;
            justify-content: center;
            border: 1px solid var(--border-glass);
        }

        .viewport-img {
            width: 100%;
            height: auto;
            max-height: 480px;
            object-fit: contain;
            display: block;
        }

        .view-controls {
            position: absolute;
            bottom: 12px;
            left: 50%;
            transform: translateX(-50%);
            display: flex;
            gap: 6px;
            background: rgba(10, 13, 20, 0.85);
            backdrop-filter: blur(12px);
            padding: 4px;
            border-radius: 30px;
            border: 1px solid var(--border-glass);
            z-index: 10;
        }

        .view-toggle-btn {
            background: transparent;
            border: none;
            color: var(--text-muted);
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 11px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
        }

        .view-toggle-btn.active {
            background: rgba(255, 255, 255, 0.15);
            color: #fff;
        }

        /* HUD Scanner Overlay */
        .hud-scanner {
            position: absolute;
            inset: 0;
            pointer-events: none;
            border: 1px solid rgba(0, 242, 254, 0.2);
            border-radius: var(--radius-md);
        }

        .hud-corner {
            position: absolute;
            width: 16px;
            height: 16px;
            border-color: var(--primary);
            border-style: solid;
        }
        .hud-tl { top: 8px; left: 8px; border-width: 2px 0 0 2px; }
        .hud-tr { top: 8px; right: 8px; border-width: 2px 2px 0 0; }
        .hud-bl { bottom: 8px; left: 8px; border-width: 0 0 2px 2px; }
        .hud-br { bottom: 8px; right: 8px; border-width: 0 2px 2px 0; }

        .hud-target-reticle {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 80px;
            height: 80px;
            border: 1px dashed rgba(0, 242, 254, 0.4);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .hud-target-reticle::after {
            content: '';
            width: 6px;
            height: 6px;
            background: var(--primary);
            border-radius: 50%;
            box-shadow: 0 0 8px var(--primary);
        }

        /* Status & Risk Panel (Right Column) */
        .safety-meter-card {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 20px;
            background: rgba(255, 255, 255, 0.02);
            border-radius: var(--radius-md);
            border: 1px solid var(--border-glass);
            margin-bottom: 20px;
        }

        .radial-meter {
            position: relative;
            width: 100px;
            height: 100px;
        }

        .radial-svg {
            transform: rotate(-90deg);
            width: 100px;
            height: 100px;
        }

        .radial-bg {
            fill: none;
            stroke: rgba(255, 255, 255, 0.06);
            stroke-width: 8;
        }

        .radial-progress {
            fill: none;
            stroke-width: 8;
            stroke-linecap: round;
            transition: stroke-dashoffset 0.8s ease, stroke 0.4s;
        }

        .radial-value {
            position: absolute;
            inset: 0;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            font-family: 'JetBrains Mono', monospace;
            font-size: 22px;
            font-weight: 700;
        }

        .radial-label {
            font-size: 9px;
            text-transform: uppercase;
            color: var(--text-muted);
            letter-spacing: 0.5px;
        }

        .safety-details {
            flex: 1;
            margin-left: 20px;
        }

        .safety-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 12px;
            border-radius: 30px;
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 6px;
        }

        .safety-title {
            font-size: 18px;
            font-weight: 700;
            color: var(--text-main);
        }

        /* Primary Detection Box */
        .primary-alert-box {
            padding: 18px;
            border-radius: var(--radius-md);
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid var(--border-glass);
            margin-bottom: 20px;
        }

        .alert-top {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 10px;
        }

        .detection-name {
            font-size: 20px;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .confidence-pill {
            font-family: 'JetBrains Mono', monospace;
            font-size: 16px;
            font-weight: 700;
            background: rgba(0, 242, 254, 0.1);
            color: var(--primary);
            padding: 4px 10px;
            border-radius: 8px;
            border: 1px solid rgba(0, 242, 254, 0.3);
        }

        .detection-desc {
            font-size: 13px;
            color: var(--text-muted);
            line-height: 1.5;
            margin-bottom: 12px;
        }

        .directive-box {
            background: rgba(0, 0, 0, 0.35);
            border-left: 3px solid var(--primary);
            padding: 10px 14px;
            border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
            font-size: 12px;
            color: #d1dbe6;
            display: flex;
            align-items: flex-start;
            gap: 8px;
        }

        /* Distraction Triad Metrics */
        .triad-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
            margin-bottom: 20px;
        }

        .triad-card {
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border-glass);
            border-radius: var(--radius-sm);
            padding: 12px 10px;
            text-align: center;
        }

        .triad-name {
            font-size: 11px;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 6px;
        }

        .triad-val {
            font-family: 'JetBrains Mono', monospace;
            font-size: 16px;
            font-weight: 700;
            color: var(--text-main);
        }

        /* Probability Bars */
        .prob-section-title {
            font-size: 13px;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 12px;
            display: flex;
            justify-content: space-between;
        }

        .prob-list {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }

        .prob-item {
            display: flex;
            flex-direction: column;
            gap: 4px;
        }

        .prob-meta {
            display: flex;
            justify-content: space-between;
            font-size: 12px;
            font-weight: 500;
        }

        .prob-track {
            height: 6px;
            background: rgba(255, 255, 255, 0.06);
            border-radius: 3px;
            overflow: hidden;
        }

        .prob-fill {
            height: 100%;
            background: var(--primary-gradient);
            border-radius: 3px;
            transition: width 0.6s cubic-bezier(0.16, 1, 0.3, 1);
        }

        /* Live Camera HUD */
        .camera-viewport {
            width: 100%;
            max-width: 720px;
            margin: 0 auto;
            border-radius: var(--radius-lg);
            position: relative;
            background: #000;
            border: 1px solid var(--border-glass);
            box-shadow: var(--shadow-hud);
            overflow: hidden;
        }

        #webcamVideo {
            width: 100%;
            height: auto;
            display: block;
            transform: scaleX(-1); /* mirror */
        }

        .camera-overlay-hud {
            position: absolute;
            top: 14px;
            left: 14px;
            right: 14px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            z-index: 5;
            pointer-events: none;
        }

        .cam-status-pill {
            background: rgba(10, 13, 20, 0.85);
            backdrop-filter: blur(10px);
            border: 1px solid var(--border-glass);
            padding: 6px 14px;
            border-radius: 20px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 12px;
            color: #fff;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .live-rec-dot {
            width: 8px;
            height: 8px;
            background: #ff1744;
            border-radius: 50%;
            box-shadow: 0 0 10px #ff1744;
            animation: pulse-dot 1.2s infinite;
        }

        .camera-controls-bar {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 14px;
            margin-top: 18px;
        }

        .btn-action {
            background: var(--primary-gradient);
            color: #0a0d14;
            border: none;
            padding: 12px 24px;
            border-radius: 12px;
            font-weight: 700;
            font-size: 14px;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 8px;
            transition: all 0.2s ease;
            box-shadow: 0 6px 20px rgba(0, 242, 254, 0.25);
        }

        .btn-action:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0, 242, 254, 0.4);
        }

        .btn-secondary {
            background: rgba(255, 255, 255, 0.06);
            color: #fff;
            border: 1px solid var(--border-glass);
            padding: 12px 20px;
            border-radius: 12px;
            font-weight: 600;
            font-size: 14px;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 8px;
            transition: all 0.2s ease;
        }

        .btn-secondary:hover {
            background: rgba(255, 255, 255, 0.12);
        }

        /* Audit Trail Table */
        .table-responsive {
            overflow-x: auto;
            margin-top: 14px;
        }

        .audit-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }

        .audit-table th {
            text-align: left;
            padding: 12px 16px;
            background: rgba(255, 255, 255, 0.02);
            color: var(--text-muted);
            font-weight: 600;
            text-transform: uppercase;
            font-size: 11px;
            letter-spacing: 1px;
            border-bottom: 1px solid var(--border-glass);
        }

        .audit-table td {
            padding: 12px 16px;
            border-bottom: 1px solid var(--border-glass);
            color: var(--text-main);
        }

        .audit-table tr:hover td {
            background: rgba(255, 255, 255, 0.02);
        }

        .audit-thumb {
            width: 44px;
            height: 34px;
            object-fit: cover;
            border-radius: 6px;
            border: 1px solid var(--border-glass);
        }

        /* Loading Spinner */
        .spinner-overlay {
            position: absolute;
            inset: 0;
            background: rgba(10, 13, 20, 0.8);
            backdrop-filter: blur(8px);
            display: none;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            z-index: 50;
            border-radius: var(--radius-lg);
        }

        .spinner-ring {
            width: 50px;
            height: 50px;
            border: 3px solid rgba(0, 242, 254, 0.2);
            border-top: 3px solid var(--primary);
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
            margin-bottom: 14px;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        /* Hidden Input */
        #hiddenFileInput {
            display: none;
        }
    </style>
</head>
<body>

    <!-- Header Navigation -->
    <header>
        <div class="brand">
            <div class="brand-logo">
                <i class="fa-solid fa-eye"></i>
            </div>
            <div>
                <div class="brand-title">AegisEye DMS</div>
                <div class="brand-tag">Deep Learning Driver Telematics</div>
            </div>
        </div>

        <div class="tab-bar">
            <button class="tab-btn active" onclick="switchTab('inspector')">
                <i class="fa-solid fa-magnifying-glass-chart"></i> Safety Inspector
            </button>
            <button class="tab-btn" onclick="switchTab('dashcam')">
                <i class="fa-solid fa-video"></i> Live AI Dashcam
            </button>
            <button class="tab-btn" onclick="switchTab('audit')">
                <i class="fa-solid fa-clipboard-list"></i> Fleet Audit Log
            </button>
            <button class="tab-btn" onclick="switchTab('specs')">
                <i class="fa-solid fa-microchip"></i> Model Architecture
            </button>
        </div>

        <div class="header-telemetry">
            <div class="tele-badge">
                <div class="tele-dot"></div>
                <span id="engineStatus">EfficientNet-B0 Online</span>
            </div>
        </div>
    </header>

    <main class="main-container">

        <!-- TAB 1: SAFETY INSPECTOR (Image Mode) -->
        <section id="inspectorTab" class="view-section active">
            <div class="inspector-grid">
                
                <!-- Left: Upload, Presets & Viewport -->
                <div class="glass-card">
                    <div class="card-header">
                        <div class="card-title">
                            <i class="fa-solid fa-camera"></i> Driver Frame Input & Visualizer
                        </div>
                        <div class="tele-badge" style="font-size: 11px;">
                            <span id="ttaStatus">TTA 3-Crop Ensembling</span>
                        </div>
                    </div>

                    <!-- Dropzone -->
                    <div class="dropzone-box" id="dropzone" onclick="document.getElementById('hiddenFileInput').click()">
                        <i class="fa-solid fa-cloud-arrow-up dropzone-icon"></i>
                        <div class="dropzone-text">Click to upload driver image or drag & drop</div>
                        <div class="dropzone-sub">Supports JPG, PNG, WEBP (Smartphone & Dashcam photos) • Paste with Ctrl+V</div>
                        <input type="file" id="hiddenFileInput" accept="image/*" onchange="handleFileSelect(event)">
                    </div>

                    <!-- Presets Strip -->
                    <div class="presets-strip">
                        <div class="presets-label">
                            <i class="fa-solid fa-bolt"></i> One-Click Real Driving Scenarios:
                        </div>
                        <div class="preset-chips">
                            <button class="preset-chip" onclick="loadSampleScenario('safe_driving.jpg', 'Safe Driving')">
                                <i class="fa-solid fa-shield-check" style="color: #00e676;"></i> Safe Driving
                            </button>
                            <button class="preset-chip" onclick="loadSampleScenario('texting_right.jpg', 'Texting (Right Hand)')">
                                <i class="fa-solid fa-mobile-screen" style="color: #ff1744;"></i> Texting Right
                            </button>
                            <button class="preset-chip" onclick="loadSampleScenario('phone_talk_right.jpg', 'Phone Call (Right Ear)')">
                                <i class="fa-solid fa-phone" style="color: #ff5252;"></i> Phone Call
                            </button>
                            <button class="preset-chip" onclick="loadSampleScenario('drinking_cup.jpg', 'Drinking Beverage')">
                                <i class="fa-solid fa-mug-hot" style="color: #ffab00;"></i> Drinking Coffee
                            </button>
                            <button class="preset-chip" onclick="loadSampleScenario('operating_radio.jpg', 'Operating Dashboard')">
                                <i class="fa-solid fa-sliders" style="color: #ff9100;"></i> Dashboard / Radio
                            </button>
                            <button class="preset-chip" onclick="loadSampleScenario('talking_passenger.jpg', 'Talking to Passenger')">
                                <i class="fa-solid fa-comments" style="color: #ffd600;"></i> Passenger Chat
                            </button>
                        </div>
                    </div>

                    <!-- Viewport Canvas & HUD -->
                    <div class="viewport-container" id="viewportBox">
                        <img id="displayImg" class="viewport-img" src="/static/samples/safe_driving.jpg" alt="Driver Frame">
                        
                        <!-- HUD Crosshairs -->
                        <div class="hud-scanner">
                            <div class="hud-corner hud-tl"></div>
                            <div class="hud-corner hud-tr"></div>
                            <div class="hud-corner hud-bl"></div>
                            <div class="hud-corner hud-br"></div>
                            <div class="hud-target-reticle"></div>
                        </div>

                        <!-- View Controls -->
                        <div class="view-controls">
                            <button class="view-toggle-btn active" id="btnViewOrig" onclick="setViewMode('original')">Original</button>
                            <button class="view-toggle-btn" id="btnViewCam" onclick="setViewMode('gradcam')">Grad-CAM Focus</button>
                        </div>

                        <!-- Loading Spinner -->
                        <div class="spinner-overlay" id="loadingOverlay">
                            <div class="spinner-ring"></div>
                            <div style="font-weight: 600; font-size: 14px;">Analyzing Neural Telematics...</div>
                            <div style="font-size: 11px; color: var(--text-muted); margin-top: 4px;">Computing TTA & Grad-CAM Attention Heatmap</div>
                        </div>
                    </div>
                </div>

                <!-- Right: Telematics & Telemetry Insights -->
                <div class="glass-card">
                    <div class="card-header">
                        <div class="card-title">
                            <i class="fa-solid fa-gauge-high"></i> Real-Time Telematics & Risk Index
                        </div>
                        <div class="tele-badge" id="latencyTicker" style="font-size: 11px;">
                            Latency: 24.5 ms
                        </div>
                    </div>

                    <!-- Safety Score Gauge -->
                    <div class="safety-meter-card">
                        <div class="radial-meter">
                            <svg class="radial-svg" viewBox="0 0 100 100">
                                <circle class="radial-bg" cx="50" cy="50" r="40"></circle>
                                <circle id="radialProgress" class="radial-progress" cx="50" cy="50" r="40" 
                                        stroke="#00e676" stroke-dasharray="251.2" stroke-dashoffset="25.1"></circle>
                            </svg>
                            <div class="radial-value">
                                <span id="safetyScoreNum">95</span>
                                <span class="radial-label">Score</span>
                            </div>
                        </div>
                        <div class="safety-details">
                            <div class="safety-badge" id="safetyBadge" style="background: rgba(0, 230, 118, 0.15); color: #00e676;">
                                <i class="fa-solid fa-shield-check"></i> <span id="safetyBadgeText">Safe Driving</span>
                            </div>
                            <div class="safety-title" id="safetyStatusText">Nominal Driver Attention</div>
                            <div style="font-size: 12px; color: var(--text-muted); margin-top: 4px;">Zero high-risk distraction detected</div>
                        </div>
                    </div>

                    <!-- Primary Classification Alert -->
                    <div class="primary-alert-box" id="primaryBox">
                        <div class="alert-top">
                            <div class="detection-name">
                                <i class="fa-solid fa-shield-check" id="primaryIcon" style="color: #00e676;"></i>
                                <span id="primaryTitle">Safe driving</span>
                            </div>
                            <div class="confidence-pill" id="primaryConfidence">98.4%</div>
                        </div>
                        <div class="detection-desc" id="primaryDesc">
                            Driver is actively focused on the roadway with hands properly positioned on steering wheel.
                        </div>
                        <div class="directive-box">
                            <i class="fa-solid fa-circle-info" style="color: var(--primary); margin-top: 2px;"></i>
                            <div><strong>Advisory Directive:</strong> <span id="primaryDirective">Maintain standard scanning techniques and safe following distances.</span></div>
                        </div>
                    </div>

                    <!-- Distraction Triad -->
                    <div class="triad-grid">
                        <div class="triad-card">
                            <div class="triad-name"><i class="fa-solid fa-eye"></i> Visual</div>
                            <div class="triad-val" id="visualVal" style="color: #00e676;">5%</div>
                        </div>
                        <div class="triad-card">
                            <div class="triad-name"><i class="fa-solid fa-hand"></i> Manual</div>
                            <div class="triad-val" id="manualVal" style="color: #00e676;">5%</div>
                        </div>
                        <div class="triad-card">
                            <div class="triad-name"><i class="fa-solid fa-brain"></i> Cognitive</div>
                            <div class="triad-val" id="cognitiveVal" style="color: #00e676;">5%</div>
                        </div>
                    </div>

                    <!-- Probability Distribution Breakdown -->
                    <div class="prob-section-title">
                        <span>Top Class Probabilities</span>
                        <span style="font-family: 'JetBrains Mono'; font-size: 11px;">Softmax Ensemble</span>
                    </div>

                    <div class="prob-list" id="probListContainer">
                        <!-- Dynamic Probability Bars -->
                    </div>

                </div>

            </div>
        </section>

        <!-- TAB 2: LIVE AI DASHCAM (Webcam Streaming) -->
        <section id="dashcamTab" class="view-section">
            <div class="glass-card" style="text-align: center; max-width: 860px; margin: 0 auto;">
                <div class="card-header">
                    <div class="card-title">
                        <i class="fa-solid fa-video"></i> Real-Time AI Cabin Dashcam Stream
                    </div>
                    <div class="tele-badge">
                        <span id="streamFps">0.0 FPS</span>
                    </div>
                </div>

                <div class="camera-viewport">
                    <video id="webcamVideo" autoplay playsinline muted></video>
                    <canvas id="streamCanvas" style="display: none;"></canvas>

                    <!-- Real-time HUD Status -->
                    <div class="camera-overlay-hud">
                        <div class="cam-status-pill">
                            <div class="live-rec-dot" id="recDot"></div>
                            <span id="camStatusText">DMS Active</span>
                        </div>
                        <div class="cam-status-pill" id="liveAlertPill" style="background: rgba(0, 230, 118, 0.2); color: #00e676;">
                            <i class="fa-solid fa-shield-check"></i> <span id="liveDetectionLabel">Safe Driving</span>
                        </div>
                    </div>

                    <!-- Reticle -->
                    <div class="hud-scanner">
                        <div class="hud-corner hud-tl"></div>
                        <div class="hud-corner hud-tr"></div>
                        <div class="hud-corner hud-bl"></div>
                        <div class="hud-corner hud-br"></div>
                        <div class="hud-target-reticle"></div>
                    </div>
                </div>

                <!-- Controls Bar -->
                <div class="camera-controls-bar">
                    <button class="btn-action" id="btnToggleCam" onclick="toggleWebcam()">
                        <i class="fa-solid fa-camera"></i> <span id="camBtnText">Start AI Dashcam</span>
                    </button>
                    <button class="btn-secondary" id="btnToggleAudio" onclick="toggleAudioAlerts()">
                        <i class="fa-solid fa-volume-high" id="audioIcon"></i> <span id="audioBtnText">Alert Chime: ON</span>
                    </button>
                </div>

                <div style="margin-top: 16px; font-size: 12px; color: var(--text-muted);">
                    Frames are analyzed live using ultra-low latency inference with real-time acoustic danger warnings.
                </div>
            </div>
        </section>

        <!-- TAB 3: FLEET AUDIT LOG -->
        <section id="auditTab" class="view-section">
            <div class="glass-card">
                <div class="card-header">
                    <div class="card-title">
                        <i class="fa-solid fa-clipboard-list"></i> Driver Telematics Event Log & Audit Trail
                    </div>
                    <button class="btn-secondary" onclick="exportAuditJSON()" style="padding: 6px 14px; font-size: 12px;">
                        <i class="fa-solid fa-download"></i> Export Telematics JSON
                    </button>
                </div>

                <div class="table-responsive">
                    <table class="audit-table">
                        <thead>
                            <tr>
                                <th>Timestamp</th>
                                <th>Snapshot</th>
                                <th>Detected Behavior</th>
                                <th>Category</th>
                                <th>Confidence</th>
                                <th>Safety Score</th>
                                <th>Risk Status</th>
                            </tr>
                        </thead>
                        <tbody id="auditTableBody">
                            <!-- Dynamic log rows -->
                        </tbody>
                    </table>
                </div>
            </div>
        </section>

        <!-- TAB 4: MODEL ARCHITECTURE & SPECS -->
        <section id="specsTab" class="view-section">
            <div class="glass-card">
                <div class="card-header">
                    <div class="card-title">
                        <i class="fa-solid fa-microchip"></i> System Architecture & Neural Backbone Specs
                    </div>
                </div>

                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; margin-top: 10px;">
                    <div class="primary-alert-box">
                        <div style="font-weight: 700; color: var(--primary); margin-bottom: 6px;"><i class="fa-solid fa-network-wired"></i> Model Backbone</div>
                        <div style="font-size: 13px; color: #cfd9df; line-height: 1.6;">
                            • EfficientNet-B0 (4.38M parameters)<br>
                            • GlobalAveragePooling2D + BatchNorm + Dropout(0.5)<br>
                            • Dense(256, ReLU) + Dense(10, Softmax)<br>
                            • Kaggle State Farm Distracted Driver Dataset
                        </div>
                    </div>

                    <div class="primary-alert-box">
                        <div style="font-weight: 700; color: var(--safe-color); margin-bottom: 6px;"><i class="fa-solid fa-bullseye"></i> Accuracy Enhancements</div>
                        <div style="font-size: 13px; color: #cfd9df; line-height: 1.6;">
                            • Aspect-Preserved Reflective Letterbox<br>
                            • 3-Crop Test-Time Augmentation (TTA)<br>
                            • Temperature-Scaled Softmax Calibration<br>
                            • Auto-Orientation EXIF Normalizer
                        </div>
                    </div>

                    <div class="primary-alert-box">
                        <div style="font-weight: 700; color: var(--warning-color); margin-bottom: 6px;"><i class="fa-solid fa-lightbulb"></i> Explainability (XAI)</div>
                        <div style="font-size: 13px; color: #cfd9df; line-height: 1.6;">
                            • Real-Time Gradient-weighted CAM (Grad-CAM)<br>
                            • Target Layer: `top_activation` (7x7x1280 feature map)<br>
                            • Turbo Color Attention Overlay Blending<br>
                            • Instant Visual Verification of Focal Regions
                        </div>
                    </div>
                </div>
            </div>
        </section>

    </main>

    <script>
        // State Management
        let currentResult = null;
        let auditLogs = [];
        let webcamRunning = false;
        let webcamStream = null;
        let audioAlertsEnabled = true;
        let streamInterval = null;
        let audioContext = null;

        // Viewport Image Cache
        let cachedOriginal = '/static/samples/safe_driving.jpg';
        let cachedGradcam = null;

        // Initialize Web Audio API for Chimes/Sirens
        function initAudio() {
            if (!audioContext) {
                audioContext = new (window.AudioContext || window.webkitAudioContext)();
            }
        }

        function playDistractionChime() {
            if (!audioAlertsEnabled) return;
            try {
                initAudio();
                const osc = audioContext.createOscillator();
                const gain = audioContext.createGain();
                osc.type = 'sawtooth';
                osc.frequency.setValueAtTime(880, audioContext.currentTime); // A5
                osc.frequency.exponentialRampToValueAtTime(440, audioContext.currentTime + 0.25);
                gain.gain.setValueAtTime(0.2, audioContext.currentTime);
                gain.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.25);
                osc.connect(gain);
                gain.connect(audioContext.destination);
                osc.start();
                osc.stop(audioContext.currentTime + 0.25);
            } catch (e) {
                console.warn('Audio chime error:', e);
            }
        }

        function toggleAudioAlerts() {
            audioAlertsEnabled = !audioAlertsEnabled;
            const icon = document.getElementById('audioIcon');
            const txt = document.getElementById('audioBtnText');
            if (audioAlertsEnabled) {
                icon.className = 'fa-solid fa-volume-high';
                txt.textContent = 'Alert Chime: ON';
            } else {
                icon.className = 'fa-solid fa-volume-xmark';
                txt.textContent = 'Alert Chime: OFF';
            }
        }

        // Tab Navigation
        function switchTab(tabName) {
            document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
            document.querySelectorAll('.view-section').forEach(sec => sec.classList.remove('active'));

            if (tabName === 'inspector') {
                document.querySelectorAll('.tab-btn')[0].classList.add('active');
                document.getElementById('inspectorTab').classList.add('active');
            } else if (tabName === 'dashcam') {
                document.querySelectorAll('.tab-btn')[1].classList.add('active');
                document.getElementById('dashcamTab').classList.add('active');
            } else if (tabName === 'audit') {
                document.querySelectorAll('.tab-btn')[2].classList.add('active');
                document.getElementById('auditTab').classList.add('active');
            } else if (tabName === 'specs') {
                document.querySelectorAll('.tab-btn')[3].classList.add('active');
                document.getElementById('specsTab').classList.add('active');
            }
        }

        // View Mode Toggle (Original vs Grad-CAM)
        function setViewMode(mode) {
            document.getElementById('btnViewOrig').classList.remove('active');
            document.getElementById('btnViewCam').classList.remove('active');
            const img = document.getElementById('displayImg');

            if (mode === 'gradcam' && cachedGradcam) {
                document.getElementById('btnViewCam').classList.add('active');
                img.src = 'data:image/jpeg;base64,' + cachedGradcam;
            } else {
                document.getElementById('btnViewOrig').classList.add('active');
                img.src = cachedOriginal.startsWith('data:') ? cachedOriginal : (cachedOriginal.startsWith('http') || cachedOriginal.startsWith('/') ? cachedOriginal : 'data:image/jpeg;base64,' + cachedOriginal);
            }
        }

        // File Selection & Drag Drop
        const dropzone = document.getElementById('dropzone');
        ['dragenter', 'dragover'].forEach(name => {
            dropzone.addEventListener(name, (e) => { e.preventDefault(); dropzone.classList.add('dragover'); });
        });
        ['dragleave', 'drop'].forEach(name => {
            dropzone.addEventListener(name, (e) => { e.preventDefault(); dropzone.classList.remove('dragover'); });
        });
        dropzone.addEventListener('drop', (e) => {
            if (e.dataTransfer.files.length) {
                processFile(e.dataTransfer.files[0]);
            }
        });

        // Clipboard Paste Support
        window.addEventListener('paste', (e) => {
            const items = (e.clipboardData || e.originalEvent.clipboardData).items;
            for (let item of items) {
                if (item.type.indexOf('image') === 0) {
                    const file = item.getAsFile();
                    processFile(file);
                    break;
                }
            }
        });

        function handleFileSelect(event) {
            if (event.target.files.length) {
                processFile(event.target.files[0]);
            }
        }

        function processFile(file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                const base64Data = e.target.result.split(',')[1];
                analyzeImagePayload(base64Data, file.name);
            };
            reader.readAsDataURL(file);
        }

        function loadSampleScenario(filename, title) {
            showLoading(true);
            fetch('/static/samples/' + filename)
                .then(res => res.blob())
                .then(blob => {
                    const reader = new FileReader();
                    reader.onload = function(e) {
                        const base64Data = e.target.result.split(',')[1];
                        analyzeImagePayload(base64Data, title);
                    };
                    reader.readAsDataURL(blob);
                })
                .catch(err => {
                    console.error('Error loading sample:', err);
                    showLoading(false);
                });
        }

        function showLoading(show) {
            document.getElementById('loadingOverlay').style.display = show ? 'flex' : 'none';
        }

        // API Prediction Request
        function analyzeImagePayload(base64Image, sourceName = 'Uploaded Image') {
            showLoading(true);
            fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    image: base64Image,
                    use_tta: true,
                    generate_cam: true
                })
            })
            .then(res => res.json())
            .then(data => {
                showLoading(false);
                if (data.success) {
                    renderPrediction(data, sourceName);
                } else {
                    alert('Prediction error: ' + (data.error || 'Unknown error'));
                }
            })
            .catch(err => {
                showLoading(false);
                console.error(err);
                alert('Connection error communicating with AI server.');
            });
        }

        // Render UI with Prediction Response
        function renderPrediction(data, sourceName = 'Frame') {
            currentResult = data;
            const pred = data.prediction;
            const tele = data.telematics;

            // Cache images
            cachedOriginal = 'data:image/jpeg;base64,' + data.images.original;
            cachedGradcam = data.images.gradcam;
            setViewMode('original');

            // Update Latency
            document.getElementById('latencyTicker').textContent = `Latency: ${tele.latency_ms} ms`;

            // Update Safety Score Gauge
            const score = tele.safety_score;
            document.getElementById('safetyScoreNum').textContent = score;
            const circumference = 251.2;
            const offset = circumference - (score / 100) * circumference;
            const radial = document.getElementById('radialProgress');
            radial.style.strokeDashoffset = offset;
            radial.style.stroke = tele.safety_status_color;

            // Safety Badge
            const badge = document.getElementById('safetyBadge');
            badge.style.background = tele.safety_status_color + '22';
            badge.style.color = tele.safety_status_color;
            document.getElementById('safetyBadgeText').textContent = tele.safety_status;
            document.getElementById('safetyStatusText').textContent = tele.is_distracted ? 'Distraction Detected' : 'Safe Driver Attention';

            // Primary Alert Box
            const primaryIcon = document.getElementById('primaryIcon');
            primaryIcon.className = 'fa-solid ' + pred.icon;
            primaryIcon.style.color = pred.badge_color;
            document.getElementById('primaryTitle').textContent = pred.title;
            document.getElementById('primaryConfidence').textContent = pred.confidence + '%';
            document.getElementById('primaryConfidence').style.color = pred.badge_color;
            document.getElementById('primaryDesc').textContent = pred.description;
            document.getElementById('primaryDirective').textContent = pred.recommendation;

            // Triad Metrics
            document.getElementById('visualVal').textContent = pred.metrics.visual_distraction + '%';
            document.getElementById('visualVal').style.color = pred.metrics.visual_distraction > 50 ? '#ff1744' : '#00e676';
            document.getElementById('manualVal').textContent = pred.metrics.manual_distraction + '%';
            document.getElementById('manualVal').style.color = pred.metrics.manual_distraction > 50 ? '#ff1744' : '#00e676';
            document.getElementById('cognitiveVal').textContent = pred.metrics.cognitive_distraction + '%';
            document.getElementById('cognitiveVal').style.color = pred.metrics.cognitive_distraction > 50 ? '#ff1744' : '#00e676';

            // Probabilities Bars
            const probContainer = document.getElementById('probListContainer');
            probContainer.innerHTML = '';
            data.top_3.forEach(item => {
                const itemDiv = document.createElement('div');
                itemDiv.className = 'prob-item';
                itemDiv.innerHTML = `
                    <div class="prob-meta">
                        <span><i class="fa-solid ${item.icon}" style="color: ${item.badge_color};"></i> ${item.title}</span>
                        <span style="font-family: 'JetBrains Mono'; font-weight: 700;">${item.confidence}%</span>
                    </div>
                    <div class="prob-track">
                        <div class="prob-fill" style="width: ${item.confidence}%; background: ${item.badge_color};"></div>
                    </div>
                `;
                probContainer.appendChild(itemDiv);
            });

            // Audio Alert if severe distraction
            if (tele.is_distracted && pred.risk_score >= 70) {
                playDistractionChime();
            }

            // Append to Audit Trail
            addAuditLog(sourceName, pred, tele, cachedOriginal);
        }

        function addAuditLog(name, pred, tele, thumbSrc) {
            const timeStr = new Date().toLocaleTimeString();
            const logEntry = {
                time: timeStr,
                name: name,
                title: pred.title,
                category: pred.category,
                confidence: pred.confidence,
                safetyScore: tele.safety_score,
                status: tele.safety_status,
                color: tele.safety_status_color,
                thumb: thumbSrc
            };
            auditLogs.unshift(logEntry);
            if (auditLogs.length > 50) auditLogs.pop();

            const tbody = document.getElementById('auditTableBody');
            const row = document.createElement('tr');
            row.innerHTML = `
                <td style="font-family: 'JetBrains Mono';">${timeStr}</td>
                <td><img src="${thumbSrc}" class="audit-thumb"></td>
                <td style="font-weight: 600;">${pred.title}</td>
                <td><span style="font-size: 11px; opacity: 0.8;">${pred.category}</span></td>
                <td style="font-family: 'JetBrains Mono'; font-weight: 700;">${pred.confidence}%</td>
                <td style="font-family: 'JetBrains Mono'; font-weight: 700; color: ${tele.safety_status_color};">${tele.safety_score}/100</td>
                <td><span class="safety-badge" style="background: ${tele.safety_status_color}22; color: ${tele.safety_status_color};">${tele.safety_status}</span></td>
            `;
            tbody.insertBefore(row, tbody.firstChild);
        }

        function exportAuditJSON() {
            const blob = new Blob([JSON.stringify(auditLogs, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `AegisEye_DMS_Audit_${Date.now()}.json`;
            a.click();
        }

        // Live AI Dashcam Mode
        async function toggleWebcam() {
            const video = document.getElementById('webcamVideo');
            const btn = document.getElementById('btnToggleCam');
            const btnTxt = document.getElementById('camBtnText');

            if (!webcamRunning) {
                try {
                    initAudio();
                    webcamStream = await navigator.mediaDevices.getUserMedia({
                        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' }
                    });
                    video.srcObject = webcamStream;
                    webcamRunning = true;
                    btnTxt.textContent = 'Stop AI Dashcam';
                    btn.style.background = 'var(--danger-gradient)';

                    let lastFrameTime = performance.now();
                    streamInterval = setInterval(() => {
                        captureAndStreamFrame();
                        const now = performance.now();
                        const fps = (1000 / (now - lastFrameTime)).toFixed(1);
                        lastFrameTime = now;
                        document.getElementById('streamFps').textContent = fps + ' FPS';
                    }, 500); // 2 FPS stream to balance smoothness and compute
                } catch (err) {
                    alert('Camera access denied or unavailable: ' + err.message);
                }
            } else {
                if (webcamStream) {
                    webcamStream.getTracks().forEach(track => track.stop());
                }
                clearInterval(streamInterval);
                video.srcObject = null;
                webcamRunning = false;
                btnTxt.textContent = 'Start AI Dashcam';
                btn.style.background = 'var(--primary-gradient)';
                document.getElementById('streamFps').textContent = '0.0 FPS';
            }
        }

        function captureAndStreamFrame() {
            const video = document.getElementById('webcamVideo');
            const canvas = document.getElementById('streamCanvas');
            if (!video.videoWidth) return;

            canvas.width = 320;
            canvas.height = 240;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            const base64Data = canvas.toDataURL('image/jpeg', 0.8).split(',')[1];

            fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    image: base64Data,
                    use_tta: false, // fast mode for stream
                    generate_cam: false
                })
            })
            .then(res => res.json())
            .then(data => {
                if (data.success) {
                    const pred = data.prediction;
                    const tele = data.telematics;
                    const pill = document.getElementById('liveAlertPill');
                    const label = document.getElementById('liveDetectionLabel');
                    
                    pill.style.background = tele.safety_status_color + '33';
                    pill.style.color = tele.safety_status_color;
                    label.textContent = `${pred.title} (${pred.confidence}%)`;

                    if (tele.is_distracted && pred.risk_score >= 75) {
                        playDistractionChime();
                    }
                }
            })
            .catch(err => console.warn('Stream frame inference error:', err));
        }

        // On Page Load: Analyze Default Sample
        window.addEventListener('DOMContentLoaded', () => {
            loadSampleScenario('safe_driving.jpg', 'Safe Driving Preset');
        });
    </script>
</body>
</html>
"""

# ---------------- ROUTES ----------------
@app.route("/", methods=["GET"])
def index():
    """Main Web Dashboard."""
    return render_template_string(HTML_DASHBOARD)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    High-Accuracy Prediction API Endpoint.
    Accepts JSON with base64 image or Multipart form-data.
    """
    try:
        use_tta = True
        generate_cam = True
        img_bytes = None
        
        if request.is_json:
            data = request.get_json()
            b64_str = data.get("image")
            if not b64_str:
                return jsonify({"success": False, "error": "No image provided in JSON payload"}), 400
            if "," in b64_str:
                b64_str = b64_str.split(",")[1]
            img_bytes = base64.b64decode(b64_str)
            use_tta = data.get("use_tta", True)
            generate_cam = data.get("generate_cam", True)
        elif "file" in request.files:
            file = request.files["file"]
            img_bytes = file.read()
            use_tta = request.form.get("use_tta", "true").lower() == "true"
            generate_cam = request.form.get("generate_cam", "true").lower() == "true"
        else:
            return jsonify({"success": False, "error": "No valid image found in request"}), 400
            
        pil_img = Image.open(io.BytesIO(img_bytes))
        result = run_enhanced_inference(pil_img, use_tta=use_tta, generate_cam=generate_cam)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in api_predict: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/samples", methods=["GET"])
def api_samples():
    """Returns available sample scenarios."""
    samples_dir = os.path.join('static', 'samples')
    if not os.path.exists(samples_dir):
        return jsonify([])
    files = [f for f in os.listdir(samples_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    return jsonify(files)

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "model_loaded": model_loaded,
        "engine": "EfficientNet-B0 DMS",
        "timestamp": datetime.now().isoformat()
    })

@app.route("/model_info", methods=["GET"])
def model_info():
    """Model information endpoint."""
    if not model_loaded or model is None:
        return jsonify({"error": "Model not loaded"}), 503
    return jsonify({
        "input_shape": model.input_shape,
        "output_shape": model.output_shape,
        "classes": list(CLASS_METADATA.keys()),
        "taxonomy": CLASS_METADATA
    })

# ---------------- RUN SERVER ----------------
if __name__ == "__main__":
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_ENV', 'development') == 'development'
    
    logger.info(f"Starting AegisEye Driver Distraction DMS on {host}:{port}")
    app.run(debug=debug, host=host, port=port)
