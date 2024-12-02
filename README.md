# Deepfake Detective - Fake Product Video Detection Web Application

A powerful web application built with Streamlit that leverages state-of-the-art deep learning models to detect product video deepfake videos. The application provides an intuitive interface for users to analyze videos and determine their authenticity.

[Live Demo](https://deep-fake-detection-m.streamlit.app/)

---

## Key Features

### Core Functionality
- **Video Analysis**: Upload and analyze MP4 videos for deepfake detection.
- **Multi-Model Support**:
  - EfficientNetB4
  - EfficientNetB4ST
  - EfficientNetAutoAttB4
  - And more...
- **Ensemble Predictions**: Select up to three models with customizable weights.
- **Adjustable Parameters**:
  - Detection threshold (0.0 to 1.0).
  - Frame selection (0 to 100 frames).
- **Real-time Results**: Immediate feedback with confidence scores.

### Technical Features
- Advanced face extraction using BlazeFace.
- Support for multiple deep learning architectures.
- Efficient video frame processing.
- Automated model weight management.

---

## Recommended Settings
- **Model**: EfficientNetAutoAttB4ST.
- **Dataset**: DFDC.
- **Frames**: At least 50 for accurate detection.
- **Threshold**: Default values unless specific requirements exist.

---

## Installation

1. Clone the repository:
    git clone https://github.com/richisen/deepfake-detection.git
    cd deepfake-detection

2. Install dependencies:
    pip install -r requirements.txt

3. Run the aplication:
    streamlit run output.py

## Author

Richik Vivek Sen

    