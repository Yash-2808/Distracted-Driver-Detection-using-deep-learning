# Driver Distraction AI

A deep learning-based web application for detecting driver distraction behaviors in real-time.

## Overview

This Flask web application uses a TensorFlow model to classify driver behavior into 10 categories, helping identify potentially dangerous driving activities.

## Features

- 🚗 Real-time driver behavior classification
- 📊 Confidence scores for predictions
- 🎯 Top 3 predictions with probabilities
- 📱 Responsive web interface
- 🖼️ Image upload and analysis
- 📝 File information display

## Driver Behavior Classes

| Class | Description |
|-------|-------------|
| c0 | Safe driving |
| c1 | Texting – Right hand |
| c2 | Talking on phone – Right hand |
| c3 | Texting – Left hand |
| c4 | Talking on phone – Left hand |
| c5 | Operating the radio |
| c6 | Drinking |
| c7 | Reaching behind |
| c8 | Hair & makeup |
| c9 | Talking to passenger |

## Installation

### Prerequisites

- Python 3.10 or 3.11 (TensorFlow compatibility)
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Yash-2808/Distracted-Driver-Detection-using-deep-learning.git
cd driver
```

2. Create and activate virtual environment:
```bash
python -m venv tf_env_compatible
# Windows
tf_env_compatible\Scripts\activate
# macOS/Linux
source tf_env_compatible/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the model files:
- **Model**: [Download v7_plus_distracted_driver.keras (41MB)](https://github.com/Yash-2808/Distracted-Driver-Detection-using-deep-learning/releases/download/v1.0/v7_plus_distracted_driver.keras)
- **Labels**: [Download labels.pkl](https://github.com/Yash-2808/Distracted-Driver-Detection-using-deep-learning/releases/download/v1.0/labels.pkl)

5. Place the downloaded files in the project directory:
```
driver/
├── app.py
├── requirements.txt
├── v7_plus_distracted_driver.keras  ← Downloaded model
└── labels.pkl                        ← Downloaded labels
```

## Usage

### Local Development

1. Start the Flask application:
```bash
python app.py
```

2. Open your browser and navigate to:
```
http://localhost:5000
```

3. Upload an image of a driver to analyze their behavior

### Deployment on Render

1. **Fork this repository** to your GitHub account

2. **Go to [Render Dashboard](https://dashboard.render.com/)** and sign up

3. **Create a new Web Service**:
   - Click "New +" → "Web Service"
   - Connect your GitHub account
   - Select the forked repository
   - Use the following settings:
     - **Runtime**: Python 3
     - **Build Command**: `pip install -r requirements.txt`
     - **Start Command**: `python app.py`
     - **Instance Type**: Free

4. **Environment Variables** (Render automatically sets these):
   - `FLASK_ENV=production`
   - `HOST=0.0.0.0`
   - `PORT=10000`
   - `PYTHON_VERSION=3.11`

5. **Deploy**: Click "Create Web Service"

Your app will be available at: `https://your-app-name.onrender.com`

### Environment Variables

Create a `.env` file for local development (see `.env.example`):

```bash
cp .env.example .env
# Edit .env with your configuration
```

Key environment variables:
- `FLASK_ENV`: Set to `production` for deployment
- `HOST`: Set to `0.0.0.0` for deployment
- `PORT`: Render uses port 10000
- `MODEL_PATH`: Path to your model file
- `LABELS_PATH`: Path to your labels file

## API Endpoints

- `GET /` - Main web interface
- `POST /` - Upload and analyze image
- `GET /health` - Health check endpoint
- `GET /model_info` - Model information endpoint

## Model Architecture

The application uses an EfficientNet-based model trained on the State Farm Distracted Driver Detection dataset. The model processes images at 224x224 pixels and outputs probabilities for 10 driver behavior classes.

## File Structure

```
driver/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── labels.pkl            # Model class labels
├── v7_plus_distracted_driver.keras  # TensorFlow model
├── .gitignore            # Git ignore file
└── README.md             # This file
```

## Dependencies

- Flask 3.1.2
- TensorFlow 2.20.0
- NumPy 2.4.1
- Pillow 12.1.0

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License.

## Acknowledgments

- State Farm Distracted Driver Detection Dataset
- TensorFlow and Keras teams
- Flask web framework
