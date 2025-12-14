# Accident Detection Using EfficientNetB1 + Transformer

This project implements an Accident Detection System using a hybrid architecture:

- EfficientNetB1 as a frame-level feature extractor

- Transformer-based temporal head to model motion patterns across time

- Real-time prediction overlay on video streams

- Adjustable FPS sampling, sequence length, and prediction stride

The system reads a video, extracts features from sampled frames, feeds a temporal sequence into a transformer model, and outputs an Accident Probability at each timestep.

## 🚀 Features

- Video sampling at configurable FPS

- Frame-level preprocessing matching training pipeline

- EfficientNetB1 backbone with ImageNet weights

- Transformer temporal model for sequence classification

- Real-time probability overlay on video frames

- Easy-to-run inference script *(inference.py)*

## 📁 Project Structure
├── inference.py

├── transformer_temporal_head.keras

├── Accident.ipynb (training / experimentation notebook)

├── data/

├    └── raw/ (example video files)

└── README.md

## 🛠️ Requirements

Install dependencies:

    pip install -r requirements.txt

## 📦 Dataset & Model Files

Due to GitHub file size limits, full-resolution videos and trained model
weights are hosted on Google Drive.

🔗 **Google Drive Folder:**
https://drive.google.com/drive/folders/1pHDF4U-Nm0WZssbd0AEjlN7bQxdkAjZH?usp=drive_link

### Folder Structure
- `models/` → transformer_temporal_head.keras  
- `videos/` → accident video clips

## ▶️ How to Run

1. Place your video inside data/raw/.

2. Update the VARIABLES section in inference.py:

        VIDEO_PATH = "data/raw/your_video.mp4"
        MODEL_WEIGHTS = "transformer_temporal_head.keras"
        SEQ_LEN = 48
        TARGET_FPS = 16

3. Run:

        python inference.py

The script will:

- Extract frames

- Preprocess them

- Run inference

- Show accident probability overlays

Press **Q** to quit the video viewer.

## 🧠 Model Details

### EfficientNetB1 Backbone

- Used for frame-level embeddings

- Outputs a 1280-dim feature vector

### Transformer Temporal Head

- Accepts shape: (None, SEQ_LEN, feat_dim)

- Learns accident dynamics over time

## 📊 Output

Each displayed frame includes:

- Accident Probability (0–1)

- Green box → safe

- Red box → above threshold (THRESH = 0.5)

## 🔧 Customization

You can adjust in USER VARIABLES:

- SEQ_LEN – temporal window

- TARGET_FPS – sampling rate

- PRED_EVERY – run transformer less often for speed

- DISPLAY_STRIDE – display fewer frames

## 📜 License

This project is licensed under the **MIT License**.

You are free to use, modify, and distribute the work.

## ✨ Author

Developed by Ponnada Kartikeya Prahit, Pratyay Patra, Komma Pallavi, Shrishti Kumari

For academic, research, and real-time accident detection experimentation.
