# AegisEye DMS – Driver Distraction AI & Safety Telematics

A production-grade, deep learning-powered Driver Monitoring System (DMS) and Safety Telematics platform designed for real-time distracted driving detection, explainable visual attention mapping (Grad-CAM), and proactive accident prevention.

---

## 🚀 Key Highlights & New Capabilities

- 🛡️ **Enterprise Cabin DMS HUD**: Sleek glassmorphism telemetry dashboard with dark mode, interactive crosshair scanners, and real-time risk gauges.
- 🎯 **High-Accuracy Vision Engine**:
  - **Aspect-Preserved Letterbox & Reflective Padding**: Eliminates distortion of driver gestures, steering postures, and smartphone grips.
  - **3-Crop Test-Time Augmentation (TTA)**: Multi-scale ensembling for robust real-world generalization across camera angles.
  - **EXIF Auto-Orientation Normalization**: Corrects rotated smartphone and dashcam JPEG captures automatically.
  - **Calibrated Softmax Probabilities**: Temperature-scaled probability smoothing for reliable confidence scores.
- 🔍 **Explainable AI (Grad-CAM Attention Mapping)**:
  - Generates high-resolution gradient-weighted class activation heatmaps (`top_activation` layer).
  - Turbo colormap overlay reveals precisely what visual cues (hands, phone, cup, radio) triggered the classification.
- 📹 **Live AI Dashcam Stream**:
  - Real-time webcam integration with frame streaming, live FPS counter, and dynamic cabin HUD.
  - Web Audio API acoustic warning chime alerts drivers when critical distraction thresholds are breached.
- 📊 **Safety Risk Telematics & Taxonomy**:
  - Computes composite **Driver Safety Index (0–100)**.
  - Distraction Triad breakdown: Visual, Manual, and Cognitive distraction metrics.
  - Context-aware actionable safety directives.
- ⚡ **Fleet Audit Trail & Export**:
  - Session event logging table with instant image snapshots, timestamps, and confidence scores.
  - 1-Click JSON telemetry export for safety compliance auditing.
- 🧪 **One-Click Real Driving Scenarios**:
  - Built-in presets for Safe Driving, Texting, Phone Calls, Drinking, Dashboard Controls, and Passenger Conversations.

---

## 🚗 Driver Behavior Classes & Risk Matrix

| Class | Behavior Description | Risk Level | Distraction Type |
|-------|----------------------|------------|------------------|
| **c0** | Safe Driving | 🟢 Nominal (5%) | Focused Roadway Gaze |
| **c1** | Texting – Right Hand | 🔴 Critical (96%) | Manual & Visual |
| **c2** | Talking on Phone – Right Hand | 🔴 High (78%) | Manual & Cognitive |
| **c3** | Texting – Left Hand | 🔴 Critical (96%) | Manual & Visual |
| **c4** | Talking on Phone – Left Hand | 🔴 High (78%) | Manual & Cognitive |
| **c5** | Operating Infotainment / Radio | 🟠 Moderate (52%) | Manual & Visual |
| **c6** | Drinking Beverage | 🟠 Moderate (62%) | Manual |
| **c7** | Reaching Behind / Rear Cabin | 🔴 Critical (90%) | Manual & Visual |
| **c8** | Hair & Makeup Grooming | 🔴 Critical (88%) | Manual & Visual |
| **c9** | Talking to Passenger / Turning Head | 🟡 Caution (42%) | Cognitive & Visual |

---

## 🛠️ Architecture & Tech Stack

- **Model Backbone**: EfficientNet-B0 (4.38M parameters) + GlobalAveragePooling2D + BatchNorm + Dense Head
- **Backend**: Python 3.10+, Flask 3.1.2, TensorFlow 2.20, NumPy, Pillow
- **Frontend**: Responsive Modern Glassmorphism Dashboard, Web Audio API, Canvas WebRTC Stream, FontAwesome 6.5, Google Fonts (Outfit & JetBrains Mono)
- **Explainability**: Custom Keras 3 GradientTape Grad-CAM Extractor

---

## 📦 Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/Yash-2808/Distracted-Driver-Detection-using-deep-learning.git
cd driver
```

### 2. Activate Virtual Environment
```bash
# Windows
tf_env_compatible\Scripts\activate

# macOS / Linux
source tf_env_compatible/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Start the Application
```bash
python app.py
```

Open your browser and navigate to:
```
http://localhost:5000
```

---

## 🌐 REST API Documentation

### `POST /api/predict`
Accepts a base64 encoded image or multipart form upload.

**Request Payload (JSON):**
```json
{
  "image": "<base64_encoded_jpeg_or_png>",
  "use_tta": true,
  "generate_cam": true
}
```

**Response Format:**
```json
{
  "success": true,
  "prediction": {
    "label": "c1",
    "title": "Texting – Right Hand",
    "category": "High Risk Distraction",
    "severity": "critical",
    "confidence": 98.4,
    "risk_score": 96,
    "description": "Driver is typing/reading text messages using right hand...",
    "recommendation": "Immediate hazard! Put the mobile device down...",
    "metrics": {
      "visual_distraction": 95,
      "manual_distraction": 90,
      "cognitive_distraction": 88
    }
  },
  "telematics": {
    "safety_score": 12,
    "safety_status": "CRITICAL DISTRACTION",
    "latency_ms": 28.4,
    "tta_enabled": true
  },
  "top_3": [...],
  "images": {
    "original": "<base64_jpeg>",
    "gradcam": "<base64_jpeg>"
  }
}
```

### Additional Endpoints
- `GET /api/samples` – List available demo driving scenarios
- `GET /health` – Server health status and model load state
- `GET /model_info` – Model architecture dimensions and complete class taxonomy

---

## 📄 License

This project is licensed under the MIT License.
