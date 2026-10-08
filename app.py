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
    if os.getenv('FORCE_CPU', 'false').lower() == 'true':
        tf.config.set_visible_devices([], 'GPU')
except Exception:
    pass

app = Flask(__name__, static_folder='static')
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'driverguard-ai-secret-2026')
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
        "title": "Safe & Attentive Driving",
        "category": "Nominal Focus",
        "severity": "safe",
        "risk_score": 4,
        "visual_distraction": 4,
        "manual_distraction": 5,
        "cognitive_distraction": 5,
        "description": "Driver is actively focused on the forward roadway with both hands properly positioned on steering wheel.",
        "recommendation": "Optimal posture maintained. Continue standard highway scanning techniques.",
        "badge_color": "#00e676",
        "icon": "fa-shield-check"
    },
    "c1": {
        "title": "Texting – Right Hand",
        "category": "Critical Distraction",
        "severity": "critical",
        "risk_score": 96,
        "visual_distraction": 96,
        "manual_distraction": 92,
        "cognitive_distraction": 90,
        "description": "Driver is typing/reading messages using right hand, diverting visual gaze and tactile control.",
        "recommendation": "Critical Hazard! Immediately stow mobile phone and return right hand to steering wheel.",
        "badge_color": "#ff1744",
        "icon": "fa-mobile-screen-button"
    },
    "c2": {
        "title": "Talking on Phone – Right Hand",
        "category": "High Risk Distraction",
        "severity": "danger",
        "risk_score": 78,
        "visual_distraction": 50,
        "manual_distraction": 88,
        "cognitive_distraction": 82,
        "description": "Driver holding phone to right ear with one hand off the steering control.",
        "recommendation": "Switch to integrated hands-free Bluetooth audio or park safely before calling.",
        "badge_color": "#ff5252",
        "icon": "fa-phone-volume"
    },
    "c3": {
        "title": "Texting – Left Hand",
        "category": "Critical Distraction",
        "severity": "critical",
        "risk_score": 96,
        "visual_distraction": 96,
        "manual_distraction": 92,
        "cognitive_distraction": 90,
        "description": "Driver is operating smartphone with left hand, significantly compromising reaction times and steering control.",
        "recommendation": "Critical Hazard! Put device away and refocus visual gaze on vehicle travel path.",
        "badge_color": "#ff1744",
        "icon": "fa-mobile-screen-button"
    },
    "c4": {
        "title": "Talking on Phone – Left Hand",
        "category": "High Risk Distraction",
        "severity": "danger",
        "risk_score": 78,
        "visual_distraction": 50,
        "manual_distraction": 88,
        "cognitive_distraction": 82,
        "description": "Driver holding phone to left ear, reducing peripheral awareness and rapid steering capability.",
        "recommendation": "Utilize hands-free voice commands or postpone conversation until vehicle is parked.",
        "badge_color": "#ff5252",
        "icon": "fa-phone-volume"
    },
    "c5": {
        "title": "Operating Radio & Console",
        "category": "Moderate Distraction",
        "severity": "warning",
        "risk_score": 54,
        "visual_distraction": 72,
        "manual_distraction": 68,
        "cognitive_distraction": 42,
        "description": "Driver interacting with center dashboard, climate dials, or infotainment touchscreen.",
        "recommendation": "Use steering-wheel mounted media controls or preset audio playlists before driving.",
        "badge_color": "#ff9100",
        "icon": "fa-sliders"
    },
    "c6": {
        "title": "Drinking Beverage",
        "category": "Moderate Distraction",
        "severity": "warning",
        "risk_score": 62,
        "visual_distraction": 48,
        "manual_distraction": 76,
        "cognitive_distraction": 35,
        "description": "Driver consuming beverage with one-handed steering control and temporary obstruction of vision.",
        "recommendation": "Consume drinks only at red lights/full stops or secure beverage firmly in cup holder.",
        "badge_color": "#ffab00",
        "icon": "fa-mug-hot"
    },
    "c7": {
        "title": "Reaching to Rear Cabin",
        "category": "Severe Distraction",
        "severity": "critical",
        "risk_score": 92,
        "visual_distraction": 94,
        "manual_distraction": 95,
        "cognitive_distraction": 72,
        "description": "Driver twisting torso and reaching behind front seats; extreme risk of involuntary lane departure.",
        "recommendation": "Never reach behind while in motion. Pull onto highway shoulder or parking spot first.",
        "badge_color": "#ff1744",
        "icon": "fa-hand-back-fist"
    },
    "c8": {
        "title": "Hair & Makeup Grooming",
        "category": "Severe Distraction",
        "severity": "critical",
        "risk_score": 89,
        "visual_distraction": 94,
        "manual_distraction": 90,
        "cognitive_distraction": 65,
        "description": "Driver looking into vanity mirror or grooming hair/makeup during vehicle transit.",
        "recommendation": "Perform personal grooming before driving or when vehicle is safely stationary.",
        "badge_color": "#ff1744",
        "icon": "fa-wand-magic-sparkles"
    },
    "c9": {
        "title": "Talking to Passenger",
        "category": "Mild-Moderate Distraction",
        "severity": "caution",
        "risk_score": 42,
        "visual_distraction": 64,
        "manual_distraction": 15,
        "cognitive_distraction": 55,
        "description": "Driver turning head toward passenger seat, intermittently taking eyes off the roadway.",
        "recommendation": "Maintain eyes forward on the road; speak without turning head away from traffic.",
        "badge_color": "#ffd600",
        "icon": "fa-comments"
    }
}

CLASS_DETAILS = {k: v["title"] for k, v in CLASS_METADATA.items()}

# ---------------- MODEL INITIALIZATION ----------------
def initialize_model():
    """Load model, labels, and initialize Grad-CAM extractor."""
    global model, eff_feature_model, labels, id_to_label, IMG_SIZE, model_loaded
    
    logger.info("Initializing DriverGuard AI Vision Engine...")
    
    # Load labels
    try:
        if os.path.exists(LABELS_PATH):
            with open(LABELS_PATH, 'rb') as f:
                labels = pickle.load(f)
            id_to_label = {v: k for k, v in labels.items()}
            logger.info(f"Labels loaded: {labels}")
        else:
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
        logger.info(f"Keras model loaded successfully. Input shape: {input_shape}")
        
        # Build Grad-CAM feature extractor
        try:
            eff = model.get_layer('efficientnetb0')
            last_conv_layer = eff.get_layer('top_activation')
            eff_feature_model = tf.keras.Model(inputs=eff.input, outputs=[last_conv_layer.output, eff.output])
            logger.info("Grad-CAM feature extractor built successfully.")
        except Exception as cam_err:
            logger.warning(f"Grad-CAM feature model could not be initialized: {cam_err}")
            eff_feature_model = None
            
        # Warm up model
        dummy = np.zeros((1, IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.float32)
        _ = model.predict(dummy, verbose=0)
        logger.info("DriverGuard AI Engine warm-up completed.")
        
        model_loaded = True
        return True
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False

# Initialize on startup
initialize_model()

# ---------------- PREPROCESSING & ACCURACY PIPELINE ----------------
def pad_and_resize_aspect_preserved(image, target_size=(224, 224)):
    """Preserves driver cabin aspect ratio with dark vehicle-interior letterbox padding."""
    orig_w, orig_h = image.size
    target_w, target_h = target_size
    
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    
    resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    padded = Image.new("RGB", target_size, (16, 20, 28))
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    padded.paste(resized, (paste_x, paste_y))
    return padded

def generate_tta_variants(pil_img, target_size=(224, 224)):
    """Generates multi-scale aspect-preserved crops for Test-Time Augmentation."""
    variants = []
    
    # 1. Base aspect-preserved padded frame
    base_padded = pad_and_resize_aspect_preserved(pil_img, target_size)
    variants.append(np.array(base_padded, dtype=np.float32))
    
    # 2. Driver cabin center crop
    w, h = pil_img.size
    min_dim = min(w, h)
    left = (w - min_dim) // 2
    top = (h - min_dim) // 2
    center_cropped = pil_img.crop((left, top, left + min_dim, top + min_dim)).resize(target_size, Image.Resampling.LANCZOS)
    variants.append(np.array(center_cropped, dtype=np.float32))
    
    # 3. Gesture zoom crop (92%)
    crop_w, crop_h = int(w * 0.92), int(h * 0.92)
    left = (w - crop_w) // 2
    top = (h - crop_h) // 2
    zoom_crop = pil_img.crop((left, top, left + crop_w, top + crop_h)).resize(target_size, Image.Resampling.LANCZOS)
    variants.append(np.array(zoom_crop, dtype=np.float32))
    
    batch = np.stack(variants, axis=0)
    return batch, base_padded

def compute_gradcam_heatmap(img_array, target_class_idx):
    """Computes high-contrast Grad-CAM attention heatmap."""
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
            
        hm_img = Image.fromarray((heatmap * 255).astype(np.uint8)).resize((224, 224), Image.Resampling.BICUBIC)
        return np.array(hm_img, dtype=np.float32) / 255.0
    except Exception as e:
        logger.warning(f"Grad-CAM error: {e}")
        return None

def colormap_turbo(val):
    """Vectorized high-definition Turbo colormap."""
    x = np.clip(val, 0, 1)
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0, 1)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0, 1)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0, 1)
    return np.stack([r, g, b], axis=-1)

def generate_cam_overlay(base_img_pil, heatmap_2d):
    """Blends Grad-CAM heatmap over driver image with smooth gaussian diffusion."""
    if heatmap_2d is None:
        return None
    try:
        color_hm = (colormap_turbo(heatmap_2d) * 255).astype(np.uint8)
        hm_pil = Image.fromarray(color_hm).resize(base_img_pil.size, Image.Resampling.BICUBIC)
        hm_pil = hm_pil.filter(ImageFilter.GaussianBlur(radius=3))
        blended = Image.blend(base_img_pil, hm_pil, alpha=0.48)
        
        buffered = io.BytesIO()
        blended.save(buffered, format="JPEG", quality=92)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception as e:
        logger.error(f"Error generating CAM overlay: {e}")
        return None

# ---------------- INFERENCE ENGINE ----------------
def run_enhanced_inference(pil_img, use_tta=True, generate_cam=True):
    start_time = time.time()
    
    # EXIF auto-orientation
    pil_img = ImageOps.exif_transpose(pil_img)
    pil_img = pil_img.convert("RGB")
    
    # Inference with TTA
    if use_tta:
        batch, base_padded = generate_tta_variants(pil_img, IMG_SIZE)
        raw_preds = model.predict(batch, verbose=0)
        weights = np.array([0.50, 0.25, 0.25]).reshape(3, 1)
        avg_preds = np.sum(raw_preds * weights, axis=0)
    else:
        base_padded = pad_and_resize_aspect_preserved(pil_img, IMG_SIZE)
        single_arr = np.expand_dims(np.array(base_padded, dtype=np.float32), axis=0)
        avg_preds = model.predict(single_arr, verbose=0)[0]
        
    # Temperature calibrated probabilities
    temperature = 0.95
    exp_preds = np.exp(np.log(np.clip(avg_preds, 1e-7, 1.0)) / temperature)
    calibrated_probs = exp_preds / np.sum(exp_preds)
    
    # Top prediction
    top_idx = int(np.argmax(calibrated_probs))
    top_label = id_to_label.get(top_idx, f"c{top_idx}")
    top_meta = CLASS_METADATA.get(top_label, CLASS_METADATA["c0"])
    top_conf = round(float(calibrated_probs[top_idx]) * 100, 1)
    
    # Full probability distribution
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
            "risk_score": meta.get("risk_score", 50),
            "badge_color": meta.get("badge_color", "#4facfe"),
            "icon": meta.get("icon", "fa-circle-dot")
        })
        
    all_predictions.sort(key=lambda x: x["confidence"], reverse=True)
    top_3 = all_predictions[:3]
    
    # Weighted Telematics Risk Score
    overall_risk = sum((p["confidence"] / 100.0) * p["risk_score"] for p in all_predictions)
    overall_safety_score = max(0, min(100, round(100 - overall_risk)))
    
    if overall_safety_score >= 80:
        safety_status = "SAFE & ATTENTIVE"
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
        
    # Grad-CAM Heatmap
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
            "engine": "DriverGuard AI Neural Core v8.0"
        },
        "top_3": top_3,
        "all_predictions": all_predictions,
        "images": {
            "original": orig_base64,
            "gradcam": cam_base64
        }
    }

# ---------------- HTML TEMPLATE ----------------
HTML_DASHBOARD = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DriverGuard AI – Intelligent Cabin Safety & Telematics</title>
    
    <!-- Fonts & Icons -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
    
    <style>
        :root {
            --bg-base: #080a0f;
            --bg-surface: rgba(14, 18, 28, 0.78);
            --bg-surface-elevated: rgba(22, 28, 44, 0.88);
            --border-subtle: rgba(255, 255, 255, 0.08);
            --border-glow: rgba(0, 242, 254, 0.35);
            
            --accent-cyan: #00f2fe;
            --accent-blue: #4facfe;
            --grad-primary: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
            --grad-safe: linear-gradient(135deg, #00e676 0%, #00b0ff 100%);
            --grad-warning: linear-gradient(135deg, #ff9100 0%, #ff5252 100%);
            --grad-danger: linear-gradient(135deg, #ff1744 0%, #f50057 100%);
            
            --color-safe: #00e676;
            --color-warning: #ff9100;
            --color-danger: #ff1744;
            
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --text-tertiary: #64748b;
            
            --radius-xl: 22px;
            --radius-lg: 16px;
            --radius-md: 10px;
            --radius-sm: 6px;
            --shadow-glass: 0 25px 60px rgba(0, 0, 0, 0.6), 0 0 35px rgba(0, 242, 254, 0.04);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
            background-color: var(--bg-base);
            color: var(--text-primary);
            min-height: 100vh;
            overflow-x: hidden;
            background-image: 
                radial-gradient(circle at 15% 15%, rgba(0, 242, 254, 0.06) 0%, transparent 45%),
                radial-gradient(circle at 85% 85%, rgba(79, 172, 254, 0.06) 0%, transparent 45%),
                radial-gradient(circle at 50% 50%, rgba(255, 23, 68, 0.02) 0%, transparent 50%),
                linear-gradient(rgba(8, 10, 15, 0.96), rgba(8, 10, 15, 0.96)),
                repeating-linear-gradient(0deg, transparent, transparent 48px, rgba(255, 255, 255, 0.012) 48px, rgba(255, 255, 255, 0.012) 49px);
        }

        /* Top Header */
        header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 16px 36px;
            background: rgba(8, 10, 15, 0.85);
            backdrop-filter: blur(24px);
            border-bottom: 1px solid var(--border-subtle);
            position: sticky;
            top: 0;
            z-index: 100;
        }

        .brand-container {
            display: flex;
            align-items: center;
            gap: 14px;
        }

        .brand-shield {
            width: 44px;
            height: 44px;
            border-radius: 12px;
            background: var(--grad-primary);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 22px;
            color: #080a0f;
            box-shadow: 0 0 25px rgba(0, 242, 254, 0.45);
        }

        .brand-name {
            font-size: 22px;
            font-weight: 800;
            letter-spacing: -0.6px;
            background: linear-gradient(120deg, #ffffff 0%, #cbd5e1 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .brand-subtitle {
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1.6px;
            color: var(--accent-cyan);
            font-family: 'JetBrains Mono', monospace;
        }

        /* Navigation Tab Pill Bar */
        .tab-nav {
            display: flex;
            gap: 6px;
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid var(--border-subtle);
            padding: 5px;
            border-radius: 14px;
        }

        .tab-item {
            background: transparent;
            border: none;
            color: var(--text-secondary);
            padding: 9px 20px;
            border-radius: 10px;
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 8px;
            transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
        }

        .tab-item:hover {
            color: var(--text-primary);
            background: rgba(255, 255, 255, 0.05);
        }

        .tab-item.active {
            background: var(--grad-primary);
            color: #080a0f;
            box-shadow: 0 4px 18px rgba(0, 242, 254, 0.3);
        }

        .telemetry-status-box {
            display: flex;
            align-items: center;
            gap: 16px;
        }

        .live-indicator {
            display: flex;
            align-items: center;
            gap: 8px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 12px;
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid var(--border-subtle);
            padding: 6px 14px;
            border-radius: 30px;
        }

        .live-pulse {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--color-safe);
            box-shadow: 0 0 12px var(--color-safe);
            animation: pulseGlow 1.8s infinite;
        }

        @keyframes pulseGlow {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.35; transform: scale(0.8); }
        }

        /* Main Container */
        .main-wrapper {
            max-width: 1440px;
            margin: 28px auto;
            padding: 0 28px;
        }

        .tab-pane {
            display: none;
        }

        .tab-pane.active {
            display: block;
            animation: paneFadeIn 0.3s ease;
        }

        @keyframes paneFadeIn {
            from { opacity: 0; transform: translateY(8px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* 2-Column Inspector Layout */
        .inspector-layout {
            display: grid;
            grid-template-columns: 1.15fr 0.85fr;
            gap: 24px;
        }

        @media (max-width: 1040px) {
            .inspector-layout {
                grid-template-columns: 1fr;
            }
        }

        /* Glass Panel */
        .glass-panel {
            background: var(--bg-surface);
            backdrop-filter: blur(28px);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-xl);
            padding: 24px;
            box-shadow: var(--shadow-glass);
            position: relative;
            overflow: hidden;
        }

        .glass-panel::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.18), transparent);
        }

        .panel-heading {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 20px;
        }

        .panel-title {
            font-size: 17px;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 10px;
            color: var(--text-primary);
        }

        .panel-title i {
            color: var(--accent-cyan);
        }

        /* Dropzone Component */
        .upload-dropzone {
            border: 2px dashed rgba(255, 255, 255, 0.12);
            border-radius: var(--radius-lg);
            padding: 30px 20px;
            text-align: center;
            background: rgba(255, 255, 255, 0.015);
            cursor: pointer;
            transition: all 0.25s ease;
            position: relative;
        }

        .upload-dropzone:hover, .upload-dropzone.dragover {
            border-color: var(--accent-cyan);
            background: rgba(0, 242, 254, 0.04);
            box-shadow: 0 0 28px rgba(0, 242, 254, 0.12);
        }

        .upload-icon {
            font-size: 40px;
            background: var(--grad-primary);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 12px;
        }

        .upload-title {
            font-size: 15px;
            font-weight: 700;
            margin-bottom: 4px;
        }

        .upload-hint {
            font-size: 12px;
            color: var(--text-secondary);
        }

        /* Preset Scenario Grid */
        .scenarios-container {
            margin-top: 18px;
        }

        .scenarios-title {
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1.2px;
            color: var(--text-tertiary);
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .scenario-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(130px, 1fr));
            gap: 8px;
        }

        .scenario-card {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-md);
            padding: 10px 12px;
            font-size: 12px;
            font-weight: 600;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 8px;
            transition: all 0.2s ease;
            color: var(--text-primary);
        }

        .scenario-card:hover {
            border-color: var(--accent-cyan);
            background: rgba(0, 242, 254, 0.08);
            transform: translateY(-2px);
        }

        /* High-Tech Viewport */
        .viewport-screen {
            margin-top: 20px;
            border-radius: var(--radius-lg);
            overflow: hidden;
            background: #000;
            position: relative;
            min-height: 330px;
            display: flex;
            align-items: center;
            justify-content: center;
            border: 1px solid var(--border-subtle);
        }

        .viewport-media {
            width: 100%;
            height: auto;
            max-height: 480px;
            object-fit: contain;
            display: block;
        }

        .hud-overlay {
            position: absolute;
            inset: 0;
            pointer-events: none;
            border: 1px solid rgba(0, 242, 254, 0.18);
            border-radius: var(--radius-lg);
        }

        .hud-corner-bracket {
            position: absolute;
            width: 18px;
            height: 18px;
            border-color: var(--accent-cyan);
            border-style: solid;
        }
        .bracket-tl { top: 10px; left: 10px; border-width: 2px 0 0 2px; }
        .bracket-tr { top: 10px; right: 10px; border-width: 2px 2px 0 0; }
        .bracket-bl { bottom: 10px; left: 10px; border-width: 0 0 2px 2px; }
        .bracket-br { bottom: 10px; right: 10px; border-width: 0 2px 2px 0; }

        .hud-reticle {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 90px;
            height: 90px;
            border: 1px dashed rgba(0, 242, 254, 0.35);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .hud-reticle::after {
            content: '';
            width: 6px;
            height: 6px;
            background: var(--accent-cyan);
            border-radius: 50%;
            box-shadow: 0 0 10px var(--accent-cyan);
        }

        .viewport-view-switcher {
            position: absolute;
            bottom: 14px;
            left: 50%;
            transform: translateX(-50%);
            display: flex;
            gap: 6px;
            background: rgba(8, 10, 15, 0.88);
            backdrop-filter: blur(14px);
            padding: 4px;
            border-radius: 30px;
            border: 1px solid var(--border-subtle);
            z-index: 10;
        }

        .view-btn {
            background: transparent;
            border: none;
            color: var(--text-secondary);
            padding: 6px 16px;
            border-radius: 20px;
            font-size: 11px;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .view-btn.active {
            background: rgba(255, 255, 255, 0.16);
            color: #fff;
        }

        /* Right Column: Telematics & Telemetry Dashboard */
        .safety-gauge-wrapper {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 22px;
            background: rgba(255, 255, 255, 0.025);
            border-radius: var(--radius-lg);
            border: 1px solid var(--border-subtle);
            margin-bottom: 20px;
        }

        .radial-gauge-container {
            position: relative;
            width: 108px;
            height: 108px;
        }

        .gauge-svg {
            transform: rotate(-90deg);
            width: 108px;
            height: 108px;
        }

        .gauge-track {
            fill: none;
            stroke: rgba(255, 255, 255, 0.06);
            stroke-width: 9;
        }

        .gauge-indicator {
            fill: none;
            stroke-width: 9;
            stroke-linecap: round;
            transition: stroke-dashoffset 0.8s ease, stroke 0.4s ease;
        }

        .gauge-center-text {
            position: absolute;
            inset: 0;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            font-family: 'JetBrains Mono', monospace;
            font-size: 24px;
            font-weight: 800;
        }

        .gauge-unit {
            font-size: 9px;
            text-transform: uppercase;
            color: var(--text-secondary);
            letter-spacing: 0.8px;
        }

        .gauge-meta-content {
            flex: 1;
            margin-left: 22px;
        }

        .status-pill-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 5px 14px;
            border-radius: 30px;
            font-size: 11px;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 8px;
        }

        .status-hero-headline {
            font-size: 20px;
            font-weight: 800;
            color: var(--text-primary);
        }

        /* Detection Spotlight Card */
        .spotlight-card {
            padding: 20px;
            border-radius: var(--radius-lg);
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid var(--border-subtle);
            margin-bottom: 20px;
        }

        .spotlight-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 10px;
        }

        .spotlight-behavior {
            font-size: 20px;
            font-weight: 800;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .confidence-tag {
            font-family: 'JetBrains Mono', monospace;
            font-size: 16px;
            font-weight: 800;
            background: rgba(0, 242, 254, 0.1);
            color: var(--accent-cyan);
            padding: 5px 12px;
            border-radius: 8px;
            border: 1px solid rgba(0, 242, 254, 0.3);
        }

        .spotlight-narrative {
            font-size: 13px;
            color: var(--text-secondary);
            line-height: 1.6;
            margin-bottom: 14px;
        }

        .action-directive-callout {
            background: rgba(0, 0, 0, 0.4);
            border-left: 3px solid var(--accent-cyan);
            padding: 12px 16px;
            border-radius: 0 var(--radius-md) var(--radius-md) 0;
            font-size: 13px;
            color: #e2e8f0;
            display: flex;
            align-items: flex-start;
            gap: 10px;
        }

        /* Triad Metrics */
        .distraction-triad-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 12px;
            margin-bottom: 20px;
        }

        .triad-box {
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-md);
            padding: 14px 10px;
            text-align: center;
        }

        .triad-type {
            font-size: 11px;
            color: var(--text-tertiary);
            text-transform: uppercase;
            font-weight: 700;
            letter-spacing: 0.6px;
            margin-bottom: 6px;
        }

        .triad-percentage {
            font-family: 'JetBrains Mono', monospace;
            font-size: 17px;
            font-weight: 800;
        }

        /* Class Distribution Breakdown */
        .distribution-heading {
            font-size: 12px;
            font-weight: 700;
            color: var(--text-tertiary);
            text-transform: uppercase;
            letter-spacing: 1.2px;
            margin-bottom: 14px;
            display: flex;
            justify-content: space-between;
        }

        .distribution-list {
            display: flex;
            flex-direction: column;
            gap: 12px;
        }

        .distribution-row {
            display: flex;
            flex-direction: column;
            gap: 5px;
        }

        .distribution-meta {
            display: flex;
            justify-content: space-between;
            font-size: 13px;
            font-weight: 600;
        }

        .distribution-bar-track {
            height: 7px;
            background: rgba(255, 255, 255, 0.06);
            border-radius: 4px;
            overflow: hidden;
        }

        .distribution-bar-fill {
            height: 100%;
            border-radius: 4px;
            transition: width 0.6s cubic-bezier(0.16, 1, 0.3, 1);
        }

        /* Live Dashcam Streaming HUD */
        .dashcam-stream-stage {
            width: 100%;
            max-width: 800px;
            margin: 0 auto;
            border-radius: var(--radius-xl);
            position: relative;
            background: #000;
            border: 1px solid var(--border-subtle);
            box-shadow: var(--shadow-glass);
            overflow: hidden;
        }

        #dashcamVideoFeed {
            width: 100%;
            height: auto;
            display: block;
            transform: scaleX(-1);
        }

        .dashcam-hud-header {
            position: absolute;
            top: 16px;
            left: 16px;
            right: 16px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            z-index: 5;
            pointer-events: none;
        }

        .hud-status-chip {
            background: rgba(8, 10, 15, 0.85);
            backdrop-filter: blur(12px);
            border: 1px solid var(--border-subtle);
            padding: 6px 16px;
            border-radius: 20px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 12px;
            color: #fff;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .rec-flasher {
            width: 8px;
            height: 8px;
            background: #ff1744;
            border-radius: 50%;
            box-shadow: 0 0 12px #ff1744;
            animation: pulseGlow 1.2s infinite;
        }

        .dashcam-action-row {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 16px;
            margin-top: 22px;
        }

        .btn-launch {
            background: var(--grad-primary);
            color: #080a0f;
            border: none;
            padding: 14px 28px;
            border-radius: var(--radius-md);
            font-weight: 800;
            font-size: 14px;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 10px;
            transition: all 0.2s ease;
            box-shadow: 0 6px 22px rgba(0, 242, 254, 0.3);
        }

        .btn-launch:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 30px rgba(0, 242, 254, 0.45);
        }

        .btn-alt {
            background: rgba(255, 255, 255, 0.06);
            color: #fff;
            border: 1px solid var(--border-subtle);
            padding: 14px 24px;
            border-radius: var(--radius-md);
            font-weight: 700;
            font-size: 14px;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 8px;
            transition: all 0.2s ease;
        }

        .btn-alt:hover {
            background: rgba(255, 255, 255, 0.12);
        }

        /* Fleet Audit Trail */
        .log-table-container {
            overflow-x: auto;
            margin-top: 16px;
        }

        .fleet-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }

        .fleet-table th {
            text-align: left;
            padding: 14px 18px;
            background: rgba(255, 255, 255, 0.02);
            color: var(--text-tertiary);
            font-weight: 700;
            text-transform: uppercase;
            font-size: 11px;
            letter-spacing: 1.2px;
            border-bottom: 1px solid var(--border-subtle);
        }

        .fleet-table td {
            padding: 14px 18px;
            border-bottom: 1px solid var(--border-subtle);
            color: var(--text-primary);
        }

        .fleet-table tr:hover td {
            background: rgba(255, 255, 255, 0.02);
        }

        .log-thumbnail {
            width: 48px;
            height: 36px;
            object-fit: cover;
            border-radius: 6px;
            border: 1px solid var(--border-subtle);
        }

        /* Loading Spinner */
        .loader-backdrop {
            position: absolute;
            inset: 0;
            background: rgba(8, 10, 15, 0.85);
            backdrop-filter: blur(10px);
            display: none;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            z-index: 50;
            border-radius: var(--radius-xl);
        }

        .loader-orbit {
            width: 52px;
            height: 52px;
            border: 3px solid rgba(0, 242, 254, 0.18);
            border-top: 3px solid var(--accent-cyan);
            border-radius: 50%;
            animation: spinOrbit 0.8s linear infinite;
            margin-bottom: 16px;
        }

        @keyframes spinOrbit {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        #filePickerInput {
            display: none;
        }
    </style>
</head>
<body>

    <!-- Header Navigation -->
    <header>
        <div class="brand-container">
            <div class="brand-shield">
                <i class="fa-solid fa-shield-halved"></i>
            </div>
            <div>
                <div class="brand-name">DriverGuard AI</div>
                <div class="brand-subtitle">Intelligent In-Cabin Safety & Telematics</div>
            </div>
        </div>

        <nav class="tab-nav">
            <button class="tab-item active" onclick="activateTab('inspector')">
                <i class="fa-solid fa-microscope"></i> Safety Inspector
            </button>
            <button class="tab-item" onclick="activateTab('dashcam')">
                <i class="fa-solid fa-video"></i> Live AI Dashcam
            </button>
            <button class="tab-item" onclick="activateTab('audit')">
                <i class="fa-solid fa-chart-line"></i> Fleet Audit Trail
            </button>
            <button class="tab-item" onclick="activateTab('specs')">
                <i class="fa-solid fa-brain"></i> Neural Core Specs
            </button>
        </nav>

        <div class="telemetry-status-box">
            <div class="live-indicator">
                <div class="live-pulse"></div>
                <span id="neuralEngineLabel">EfficientNet-B0 Online</span>
            </div>
        </div>
    </header>

    <main class="main-wrapper">

        <!-- TAB 1: SAFETY INSPECTOR -->
        <section id="paneInspector" class="tab-pane active">
            <div class="inspector-layout">
                
                <!-- Left: Upload, Scenarios & Viewport -->
                <div class="glass-panel">
                    <div class="panel-heading">
                        <div class="panel-title">
                            <i class="fa-solid fa-camera-viewfinder"></i> Driver Frame Analyzer
                        </div>
                        <div class="live-indicator" style="font-size: 11px;">
                            <span>3-Crop TTA Ensembling Active</span>
                        </div>
                    </div>

                    <!-- Dropzone -->
                    <div class="upload-dropzone" id="uploadDropzone" onclick="document.getElementById('filePickerInput').click()">
                        <i class="fa-solid fa-cloud-arrow-up upload-icon"></i>
                        <div class="upload-title">Drop driver photo here or click to browse</div>
                        <div class="upload-hint">Dashcam, smartphone, or cabin CCTV images • Paste directly with Ctrl+V</div>
                        <input type="file" id="filePickerInput" accept="image/*" onchange="onFileChosen(event)">
                    </div>

                    <!-- Realistic Driving Scenarios -->
                    <div class="scenarios-container">
                        <div class="scenarios-title">
                            <i class="fa-solid fa-bolt"></i> One-Click Real Driving Presets:
                        </div>
                        <div class="scenario-grid">
                            <button class="scenario-card" onclick="loadPresetImage('safe_driving.jpg', 'Safe Driving')">
                                <i class="fa-solid fa-shield-check" style="color: #00e676;"></i> Safe Driving
                            </button>
                            <button class="scenario-card" onclick="loadPresetImage('texting_right.jpg', 'Texting (Right Hand)')">
                                <i class="fa-solid fa-mobile-screen" style="color: #ff1744;"></i> Texting Right
                            </button>
                            <button class="scenario-card" onclick="loadPresetImage('phone_talk_right.jpg', 'Talking on Phone')">
                                <i class="fa-solid fa-phone-volume" style="color: #ff5252;"></i> Phone Call
                            </button>
                            <button class="scenario-card" onclick="loadPresetImage('drinking_cup.jpg', 'Drinking Beverage')">
                                <i class="fa-solid fa-mug-hot" style="color: #ffab00;"></i> Drinking Cup
                            </button>
                            <button class="scenario-card" onclick="loadPresetImage('operating_radio.jpg', 'Radio / Console')">
                                <i class="fa-solid fa-sliders" style="color: #ff9100;"></i> Radio Console
                            </button>
                            <button class="scenario-card" onclick="loadPresetImage('talking_passenger.jpg', 'Talking to Passenger')">
                                <i class="fa-solid fa-comments" style="color: #ffd600;"></i> Passenger Chat
                            </button>
                        </div>
                    </div>

                    <!-- Viewport Screen & HUD -->
                    <div class="viewport-screen" id="viewportScreen">
                        <img id="activeViewportImg" class="viewport-media" src="/static/samples/safe_driving.jpg" alt="Driver Analysis">
                        
                        <!-- HUD Reticle -->
                        <div class="hud-overlay">
                            <div class="hud-corner-bracket bracket-tl"></div>
                            <div class="hud-corner-bracket bracket-tr"></div>
                            <div class="hud-corner-bracket bracket-bl"></div>
                            <div class="hud-corner-bracket bracket-br"></div>
                            <div class="hud-reticle"></div>
                        </div>

                        <!-- View Controls -->
                        <div class="viewport-view-switcher">
                            <button class="view-btn active" id="btnOrigView" onclick="toggleViewMode('original')">Original Frame</button>
                            <button class="view-btn" id="btnCamView" onclick="toggleViewMode('gradcam')">Grad-CAM Focus</button>
                        </div>

                        <!-- Loading Overlay -->
                        <div class="loader-backdrop" id="loaderOverlay">
                            <div class="loader-orbit"></div>
                            <div style="font-weight: 700; font-size: 15px;">Executing DriverGuard Neural Telematics...</div>
                            <div style="font-size: 12px; color: var(--text-secondary); margin-top: 4px;">Computing TTA Multi-Crop & Grad-CAM Visual Heatmap</div>
                        </div>
                    </div>
                </div>

                <!-- Right: Telematics Metrics & Behavior Analysis -->
                <div class="glass-panel">
                    <div class="panel-heading">
                        <div class="panel-title">
                            <i class="fa-solid fa-gauge-high"></i> In-Cabin Telematics Risk Index
                        </div>
                        <div class="live-indicator" id="latencyIndicator" style="font-size: 11px;">
                            Latency: 22.4 ms
                        </div>
                    </div>

                    <!-- Driver Safety Score Radial Gauge -->
                    <div class="safety-gauge-wrapper">
                        <div class="radial-gauge-container">
                            <svg class="gauge-svg" viewBox="0 0 100 100">
                                <circle class="gauge-track" cx="50" cy="50" r="40"></circle>
                                <circle id="gaugeProgressArc" class="gauge-indicator" cx="50" cy="50" r="40" 
                                        stroke="#00e676" stroke-dasharray="251.2" stroke-dashoffset="25.1"></circle>
                            </svg>
                            <div class="gauge-center-text">
                                <span id="gaugeNumberVal">96</span>
                                <span class="gauge-unit">Safety Score</span>
                            </div>
                        </div>
                        <div class="gauge-meta-content">
                            <div class="status-pill-badge" id="safetyStatusPill" style="background: rgba(0, 230, 118, 0.15); color: #00e676;">
                                <i class="fa-solid fa-shield-check"></i> <span id="safetyStatusPillText">Safe & Attentive</span>
                            </div>
                            <div class="status-hero-headline" id="safetyHeroTitle">Nominal Driver Attention</div>
                            <div style="font-size: 12px; color: var(--text-secondary); margin-top: 4px;">Zero high-risk distraction detected</div>
                        </div>
                    </div>

                    <!-- Primary Detection Spotlight -->
                    <div class="spotlight-card">
                        <div class="spotlight-header">
                            <div class="spotlight-behavior">
                                <i class="fa-solid fa-shield-check" id="spotlightIcon" style="color: #00e676;"></i>
                                <span id="spotlightTitle">Safe & Attentive Driving</span>
                            </div>
                            <div class="confidence-tag" id="spotlightConf">98.5%</div>
                        </div>
                        <div class="spotlight-narrative" id="spotlightDesc">
                            Driver is actively focused on the forward roadway with both hands properly positioned on steering wheel.
                        </div>
                        <div class="action-directive-callout">
                            <i class="fa-solid fa-circle-info" style="color: var(--accent-cyan); margin-top: 2px;"></i>
                            <div><strong>Advisory Directive:</strong> <span id="spotlightDirective">Optimal posture maintained. Continue standard highway scanning techniques.</span></div>
                        </div>
                    </div>

                    <!-- Distraction Triad -->
                    <div class="distraction-triad-grid">
                        <div class="triad-box">
                            <div class="triad-type"><i class="fa-solid fa-eye"></i> Visual</div>
                            <div class="triad-percentage" id="triadVisual" style="color: #00e676;">4%</div>
                        </div>
                        <div class="triad-box">
                            <div class="triad-type"><i class="fa-solid fa-hand"></i> Manual</div>
                            <div class="triad-percentage" id="triadManual" style="color: #00e676;">5%</div>
                        </div>
                        <div class="triad-box">
                            <div class="triad-type"><i class="fa-solid fa-brain"></i> Cognitive</div>
                            <div class="triad-percentage" id="triadCognitive" style="color: #00e676;">5%</div>
                        </div>
                    </div>

                    <!-- Class Probability Breakdown -->
                    <div class="distribution-heading">
                        <span>Top Class Probabilities</span>
                        <span style="font-family: 'JetBrains Mono'; font-size: 11px;">Softmax Calibrated</span>
                    </div>

                    <div class="distribution-list" id="distributionContainer">
                        <!-- Dynamic Probability Rows -->
                    </div>

                </div>

            </div>
        </section>

        <!-- TAB 2: LIVE AI DASHCAM -->
        <section id="paneDashcam" class="tab-pane">
            <div class="glass-panel" style="text-align: center; max-width: 860px; margin: 0 auto;">
                <div class="panel-heading">
                    <div class="panel-title">
                        <i class="fa-solid fa-video"></i> Live AI In-Cabin Dashcam Monitor
                    </div>
                    <div class="live-indicator">
                        <span id="liveFpsTicker">0.0 FPS</span>
                    </div>
                </div>

                <div class="dashcam-stream-stage">
                    <video id="dashcamVideoFeed" autoplay playsinline muted></video>
                    <canvas id="hiddenStreamCanvas" style="display: none;"></canvas>

                    <!-- Real-time HUD Status -->
                    <div class="dashcam-hud-header">
                        <div class="hud-status-chip">
                            <div class="rec-flasher" id="recDotFlasher"></div>
                            <span id="streamStateText">DriverGuard Active</span>
                        </div>
                        <div class="hud-status-chip" id="streamAlertChip" style="background: rgba(0, 230, 118, 0.2); color: #00e676;">
                            <i class="fa-solid fa-shield-check"></i> <span id="streamBehaviorLabel">Safe Driving</span>
                        </div>
                    </div>

                    <div class="hud-overlay">
                        <div class="hud-corner-bracket bracket-tl"></div>
                        <div class="hud-corner-bracket bracket-tr"></div>
                        <div class="hud-corner-bracket bracket-bl"></div>
                        <div class="hud-corner-bracket bracket-br"></div>
                        <div class="hud-reticle"></div>
                    </div>
                </div>

                <!-- Controls -->
                <div class="dashcam-action-row">
                    <button class="btn-launch" id="btnToggleStream" onclick="handleStreamToggle()">
                        <i class="fa-solid fa-camera"></i> <span id="streamToggleLabel">Start AI Dashcam</span>
                    </button>
                    <button class="btn-alt" id="btnToggleAlarm" onclick="handleAlarmToggle()">
                        <i class="fa-solid fa-volume-high" id="alarmIcon"></i> <span id="alarmToggleLabel">Alert Chime: ON</span>
                    </button>
                </div>

                <div style="margin-top: 18px; font-size: 13px; color: var(--text-secondary);">
                    Real-time in-cabin frame analysis with low-latency neural inference and acoustic alert warnings.
                </div>
            </div>
        </section>

        <!-- TAB 3: FLEET AUDIT TRAIL -->
        <section id="paneAudit" class="tab-pane">
            <div class="glass-panel">
                <div class="panel-heading">
                    <div class="panel-title">
                        <i class="fa-solid fa-chart-line"></i> Fleet Telematics Audit Trail & Event Logs
                    </div>
                    <button class="btn-alt" onclick="downloadAuditJSON()" style="padding: 8px 18px; font-size: 12px;">
                        <i class="fa-solid fa-download"></i> Export Telematics JSON
                    </button>
                </div>

                <div class="log-table-container">
                    <table class="fleet-table">
                        <thead>
                            <tr>
                                <th>Timestamp</th>
                                <th>Snapshot</th>
                                <th>Detected Behavior</th>
                                <th>Category</th>
                                <th>Confidence</th>
                                <th>Safety Score</th>
                                <th>Status</th>
                            </tr>
                        </thead>
                        <tbody id="fleetLogTableBody">
                            <!-- Dynamic log rows -->
                        </tbody>
                    </table>
                </div>
            </div>
        </section>

        <!-- TAB 4: NEURAL CORE SPECS -->
        <section id="paneSpecs" class="tab-pane">
            <div class="glass-panel">
                <div class="panel-heading">
                    <div class="panel-title">
                        <i class="fa-solid fa-brain"></i> DriverGuard AI Neural Backbone Architecture
                    </div>
                </div>

                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(290px, 1fr)); gap: 18px; margin-top: 12px;">
                    <div class="spotlight-card">
                        <div style="font-weight: 800; color: var(--accent-cyan); margin-bottom: 8px;"><i class="fa-solid fa-network-wired"></i> Model Backbone</div>
                        <div style="font-size: 13px; color: #cbd5e1; line-height: 1.7;">
                            • EfficientNet-B0 (4,385,197 parameters)<br>
                            • GlobalAveragePooling2D + BatchNorm + Dropout(0.5)<br>
                            • Dense(256, ReLU) + Dense(10, Softmax)<br>
                            • State Farm Distracted Driver Dataset
                        </div>
                    </div>

                    <div class="spotlight-card">
                        <div style="font-weight: 800; color: var(--color-safe); margin-bottom: 8px;"><i class="fa-solid fa-crosshairs"></i> Precision Enhancements</div>
                        <div style="font-size: 13px; color: #cbd5e1; line-height: 1.7;">
                            • Aspect-Preserved Vehicle Cabin Letterbox<br>
                            • 3-Crop Test-Time Augmentation (TTA)<br>
                            • Temperature-Scaled Softmax Calibration<br>
                            • EXIF Auto-Orientation Normalization
                        </div>
                    </div>

                    <div class="spotlight-card">
                        <div style="font-weight: 800; color: var(--color-warning); margin-bottom: 8px;"><i class="fa-solid fa-eye"></i> Explainable AI (XAI)</div>
                        <div style="font-size: 13px; color: #cbd5e1; line-height: 1.7;">
                            • Real-Time Gradient-weighted CAM (Grad-CAM)<br>
                            • Target Layer: `top_activation` (7x7x1280)<br>
                            • High-Contrast Turbo Attention Heatmap<br>
                            • Visual Verification of Hand & Device Coordinates
                        </div>
                    </div>
                </div>
            </div>
        </section>

    </main>

    <script>
        // State
        let auditRecords = [];
        let isStreaming = false;
        let activeMediaStream = null;
        let alarmAudioEnabled = true;
        let streamTimer = null;
        let webAudioCtx = null;

        let activeOriginalImg = '/static/samples/safe_driving.jpg';
        let activeGradcamImg = null;

        // Audio Chime Synthesizer
        function initWebAudio() {
            if (!webAudioCtx) {
                webAudioCtx = new (window.AudioContext || window.webkitAudioContext)();
            }
        }

        function triggerDangerChime() {
            if (!alarmAudioEnabled) return;
            try {
                initWebAudio();
                const osc = webAudioCtx.createOscillator();
                const gain = webAudioCtx.createGain();
                osc.type = 'sawtooth';
                osc.frequency.setValueAtTime(880, webAudioCtx.currentTime);
                osc.frequency.exponentialRampToValueAtTime(440, webAudioCtx.currentTime + 0.28);
                gain.gain.setValueAtTime(0.22, webAudioCtx.currentTime);
                gain.gain.exponentialRampToValueAtTime(0.01, webAudioCtx.currentTime + 0.28);
                osc.connect(gain);
                gain.connect(webAudioCtx.destination);
                osc.start();
                osc.stop(webAudioCtx.currentTime + 0.28);
            } catch (e) {
                console.warn('Audio chime notice:', e);
            }
        }

        function handleAlarmToggle() {
            alarmAudioEnabled = !alarmAudioEnabled;
            const icon = document.getElementById('alarmIcon');
            const lbl = document.getElementById('alarmToggleLabel');
            if (alarmAudioEnabled) {
                icon.className = 'fa-solid fa-volume-high';
                lbl.textContent = 'Alert Chime: ON';
            } else {
                icon.className = 'fa-solid fa-volume-xmark';
                lbl.textContent = 'Alert Chime: OFF';
            }
        }

        // Tab Switching
        function activateTab(tabKey) {
            document.querySelectorAll('.tab-item').forEach(btn => btn.classList.remove('active'));
            document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));

            if (tabKey === 'inspector') {
                document.querySelectorAll('.tab-item')[0].classList.add('active');
                document.getElementById('paneInspector').classList.add('active');
            } else if (tabKey === 'dashcam') {
                document.querySelectorAll('.tab-item')[1].classList.add('active');
                document.getElementById('paneDashcam').classList.add('active');
            } else if (tabKey === 'audit') {
                document.querySelectorAll('.tab-item')[2].classList.add('active');
                document.getElementById('paneAudit').classList.add('active');
            } else if (tabKey === 'specs') {
                document.querySelectorAll('.tab-item')[3].classList.add('active');
                document.getElementById('paneSpecs').classList.add('active');
            }
        }

        // View Mode Switch (Original vs Grad-CAM)
        function toggleViewMode(mode) {
            document.getElementById('btnOrigView').classList.remove('active');
            document.getElementById('btnCamView').classList.remove('active');
            const media = document.getElementById('activeViewportImg');

            if (mode === 'gradcam' && activeGradcamImg) {
                document.getElementById('btnCamView').classList.add('active');
                media.src = 'data:image/jpeg;base64,' + activeGradcamImg;
            } else {
                document.getElementById('btnOrigView').classList.add('active');
                media.src = activeOriginalImg.startsWith('data:') ? activeOriginalImg : (activeOriginalImg.startsWith('http') || activeOriginalImg.startsWith('/') ? activeOriginalImg : 'data:image/jpeg;base64,' + activeOriginalImg);
            }
        }

        // Drag & Drop
        const dropzoneEl = document.getElementById('uploadDropzone');
        ['dragenter', 'dragover'].forEach(ev => {
            dropzoneEl.addEventListener(ev, (e) => { e.preventDefault(); dropzoneEl.classList.add('dragover'); });
        });
        ['dragleave', 'drop'].forEach(ev => {
            dropzoneEl.addEventListener(ev, (e) => { e.preventDefault(); dropzoneEl.classList.remove('dragover'); });
        });
        dropzoneEl.addEventListener('drop', (e) => {
            if (e.dataTransfer.files.length) {
                readAndAnalyzeFile(e.dataTransfer.files[0]);
            }
        });

        // Clipboard Paste
        window.addEventListener('paste', (e) => {
            const items = (e.clipboardData || e.originalEvent.clipboardData).items;
            for (let item of items) {
                if (item.type.indexOf('image') === 0) {
                    const file = item.getAsFile();
                    readAndAnalyzeFile(file);
                    break;
                }
            }
        });

        function onFileChosen(event) {
            if (event.target.files.length) {
                readAndAnalyzeFile(event.target.files[0]);
            }
        }

        function readAndAnalyzeFile(file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                const b64 = e.target.result.split(',')[1];
                sendInferenceRequest(b64, file.name);
            };
            reader.readAsDataURL(file);
        }

        function loadPresetImage(filename, title) {
            setLoading(true);
            fetch('/static/samples/' + filename)
                .then(r => r.blob())
                .then(b => {
                    const reader = new FileReader();
                    reader.onload = function(e) {
                        const b64 = e.target.result.split(',')[1];
                        sendInferenceRequest(b64, title);
                    };
                    reader.readAsDataURL(b);
                })
                .catch(err => {
                    console.error('Preset error:', err);
                    setLoading(false);
                });
        }

        function setLoading(isLoading) {
            document.getElementById('loaderOverlay').style.display = isLoading ? 'flex' : 'none';
        }

        function sendInferenceRequest(base64Payload, title = 'Uploaded Image') {
            setLoading(true);
            fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    image: base64Payload,
                    use_tta: true,
                    generate_cam: true
                })
            })
            .then(res => res.json())
            .then(data => {
                setLoading(false);
                if (data.success) {
                    updateDashboardUI(data, title);
                } else {
                    alert('Prediction Notice: ' + (data.error || 'Server error'));
                }
            })
            .catch(err => {
                setLoading(false);
                console.error(err);
                alert('Connection issue contacting DriverGuard AI server.');
            });
        }

        function updateDashboardUI(data, sourceTitle = 'Frame') {
            const pred = data.prediction;
            const tele = data.telematics;

            activeOriginalImg = 'data:image/jpeg;base64,' + data.images.original;
            activeGradcamImg = data.images.gradcam;
            toggleViewMode('original');

            // Latency
            document.getElementById('latencyIndicator').textContent = `Latency: ${tele.latency_ms} ms`;

            // Safety Score Gauge
            const score = tele.safety_score;
            document.getElementById('gaugeNumberVal').textContent = score;
            const circumference = 251.2;
            const offset = circumference - (score / 100) * circumference;
            const arc = document.getElementById('gaugeProgressArc');
            arc.style.strokeDashoffset = offset;
            arc.style.stroke = tele.safety_status_color;

            // Status Pill
            const pill = document.getElementById('safetyStatusPill');
            pill.style.background = tele.safety_status_color + '24';
            pill.style.color = tele.safety_status_color;
            document.getElementById('safetyStatusPillText').textContent = tele.safety_status;
            document.getElementById('safetyHeroTitle').textContent = tele.is_distracted ? 'Distraction Detected' : 'Safe Driver Attention';

            // Spotlight Card
            const iconEl = document.getElementById('spotlightIcon');
            iconEl.className = 'fa-solid ' + pred.icon;
            iconEl.style.color = pred.badge_color;
            document.getElementById('spotlightTitle').textContent = pred.title;
            document.getElementById('spotlightConf').textContent = pred.confidence + '%';
            document.getElementById('spotlightConf').style.color = pred.badge_color;
            document.getElementById('spotlightDesc').textContent = pred.description;
            document.getElementById('spotlightDirective').textContent = pred.recommendation;

            // Triad
            document.getElementById('triadVisual').textContent = pred.metrics.visual_distraction + '%';
            document.getElementById('triadVisual').style.color = pred.metrics.visual_distraction > 50 ? '#ff1744' : '#00e676';
            document.getElementById('triadManual').textContent = pred.metrics.manual_distraction + '%';
            document.getElementById('triadManual').style.color = pred.metrics.manual_distraction > 50 ? '#ff1744' : '#00e676';
            document.getElementById('triadCognitive').textContent = pred.metrics.cognitive_distraction + '%';
            document.getElementById('triadCognitive').style.color = pred.metrics.cognitive_distraction > 50 ? '#ff1744' : '#00e676';

            // Probability Distribution Rows
            const container = document.getElementById('distributionContainer');
            container.innerHTML = '';
            data.top_3.forEach(item => {
                const row = document.createElement('div');
                row.className = 'distribution-row';
                row.innerHTML = `
                    <div class="distribution-meta">
                        <span><i class="fa-solid ${item.icon}" style="color: ${item.badge_color};"></i> ${item.title}</span>
                        <span style="font-family: 'JetBrains Mono'; font-weight: 700;">${item.confidence}%</span>
                    </div>
                    <div class="distribution-bar-track">
                        <div class="distribution-bar-fill" style="width: ${item.confidence}%; background: ${item.badge_color};"></div>
                    </div>
                `;
                container.appendChild(row);
            });

            // Danger Chime
            if (tele.is_distracted && pred.risk_score >= 70) {
                triggerDangerChime();
            }

            // Record Log
            appendFleetLog(sourceTitle, pred, tele, activeOriginalImg);
        }

        function appendFleetLog(name, pred, tele, thumbnail) {
            const timeStr = new Date().toLocaleTimeString();
            const record = {
                time: timeStr,
                name: name,
                title: pred.title,
                category: pred.category,
                confidence: pred.confidence,
                safetyScore: tele.safety_score,
                status: tele.safety_status,
                color: tele.safety_status_color,
                thumb: thumbnail
            };
            auditRecords.unshift(record);
            if (auditRecords.length > 50) auditRecords.pop();

            const tbody = document.getElementById('fleetLogTableBody');
            const row = document.createElement('tr');
            row.innerHTML = `
                <td style="font-family: 'JetBrains Mono'; font-size: 12px;">${timeStr}</td>
                <td><img src="${thumbnail}" class="log-thumbnail"></td>
                <td style="font-weight: 700;">${pred.title}</td>
                <td><span style="font-size: 12px; color: var(--text-secondary);">${pred.category}</span></td>
                <td style="font-family: 'JetBrains Mono'; font-weight: 700;">${pred.confidence}%</td>
                <td style="font-family: 'JetBrains Mono'; font-weight: 800; color: ${tele.safety_status_color};">${tele.safety_score}/100</td>
                <td><span class="status-pill-badge" style="background: ${tele.safety_status_color}22; color: ${tele.safety_status_color};">${tele.safety_status}</span></td>
            `;
            tbody.insertBefore(row, tbody.firstChild);
        }

        function downloadAuditJSON() {
            const blob = new Blob([JSON.stringify(auditRecords, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `DriverGuard_AI_Fleet_Audit_${Date.now()}.json`;
            a.click();
        }

        // Live AI Dashcam Mode
        async function handleStreamToggle() {
            const video = document.getElementById('dashcamVideoFeed');
            const btn = document.getElementById('btnToggleStream');
            const label = document.getElementById('streamToggleLabel');

            if (!isStreaming) {
                try {
                    initWebAudio();
                    activeMediaStream = await navigator.mediaDevices.getUserMedia({
                        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' }
                    });
                    video.srcObject = activeMediaStream;
                    isStreaming = true;
                    label.textContent = 'Stop AI Dashcam';
                    btn.style.background = 'var(--grad-danger)';

                    let lastFrameTime = performance.now();
                    streamTimer = setInterval(() => {
                        captureStreamFrame();
                        const now = performance.now();
                        const fps = (1000 / (now - lastFrameTime)).toFixed(1);
                        lastFrameTime = now;
                        document.getElementById('liveFpsTicker').textContent = fps + ' FPS';
                    }, 500);
                } catch (err) {
                    alert('Camera access denied or device unavailable: ' + err.message);
                }
            } else {
                if (activeMediaStream) {
                    activeMediaStream.getTracks().forEach(t => t.stop());
                }
                clearInterval(streamTimer);
                video.srcObject = null;
                isStreaming = false;
                label.textContent = 'Start AI Dashcam';
                btn.style.background = 'var(--grad-primary)';
                document.getElementById('liveFpsTicker').textContent = '0.0 FPS';
            }
        }

        function captureStreamFrame() {
            const video = document.getElementById('dashcamVideoFeed');
            const canvas = document.getElementById('hiddenStreamCanvas');
            if (!video.videoWidth) return;

            canvas.width = 320;
            canvas.height = 240;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            const b64 = canvas.toDataURL('image/jpeg', 0.8).split(',')[1];

            fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    image: b64,
                    use_tta: false,
                    generate_cam: false
                })
            })
            .then(r => r.json())
            .then(data => {
                if (data.success) {
                    const pred = data.prediction;
                    const tele = data.telematics;
                    const chip = document.getElementById('streamAlertChip');
                    const label = document.getElementById('streamBehaviorLabel');
                    
                    chip.style.background = tele.safety_status_color + '33';
                    chip.style.color = tele.safety_status_color;
                    label.textContent = `${pred.title} (${pred.confidence}%)`;

                    if (tele.is_distracted && pred.risk_score >= 75) {
                        triggerDangerChime();
                    }
                }
            })
            .catch(e => console.warn('Stream inference notice:', e));
        }

        // On Load: Analyze Initial Preset
        window.addEventListener('DOMContentLoaded', () => {
            loadPresetImage('safe_driving.jpg', 'Safe Driving Preset');
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
    """Returns available sample scenario images."""
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
        "engine": "DriverGuard AI Neural Core v8.0",
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
    
    logger.info(f"Starting DriverGuard AI DMS on {host}:{port}")
    app.run(debug=debug, host=host, port=port)
