# 🛡️ FaceShield — Face Anti-Spoofing Detection System

Real-time face liveness detection that distinguishes **genuine human faces** from **spoofed/fake faces** (printed photos, digital screens, masks) using multi-cue computer vision analysis.

---

## ✨ Features

| Feature | Details |
|---|---|
| 🎯 Real vs Spoof Detection | Multi-layer analysis pipeline |
| 📷 Live Camera Feed | WebSocket-streamed annotated video |
| 📊 Confidence Scores | Per-face real/spoof confidence |
| 🔬 Feature Breakdown | Texture, sharpness, color, frequency |
| 📋 Detection Log | Timestamped event history |
| 💾 Snapshot Capture | Save annotated frames |
| 🌐 Web Dashboard | Access from any browser |

---

## 🧠 Detection Pipeline

```
Camera Frame
    ↓
Face Detection (DNN / Haar Cascade)
    ↓
ROI Extraction (padded bounding box)
    ↓
┌─────────────────────────────────────────┐
│  1. LBP Texture Entropy                 │
│     Real skin → high varied entropy     │
│  2. Laplacian Sharpness                 │
│     Natural faces → smooth gradients    │
│  3. Color Channel Variance              │
│     Photos/screens appear flat          │
│  4. FFT Frequency Analysis              │
│     Screens produce moiré patterns      │
│  5. Temporal Consistency                │
│     Live faces have natural motion      │
└─────────────────────────────────────────┘
    ↓
Weighted Fusion → Confidence Score
    ↓
REAL ✅  or  SPOOF 🚨
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Run the Server

```bash
cd backend
python main.py
```

### 3. Open Dashboard

Visit **http://localhost:8000** in your browser.

Press **Start Detection** to begin live analysis.

---

## 🗂️ Project Structure

```
face-spoof-detection/
├── backend/
│   ├── main.py          # FastAPI server + WebSocket
│   ├── detector.py      # Anti-spoofing logic
│   ├── models/          # (optional) DNN model weights
│   └── requirements.txt
├── frontend/
│   ├── index.html       # Dashboard UI
│   ├── style.css        # Premium dark design
│   └── app.js           # WebSocket client + UI control
└── README.md
```

---

## 🔌 API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Dashboard UI |
| `/api/status` | GET | Server & session status |
| `/api/analyze` | POST | Analyze base64 image |
| `/api/camera/start` | POST | Start server camera |
| `/api/camera/stop` | POST | Stop server camera |
| `/api/log` | GET | Recent detection log |
| `/ws/detect` | WS | Real-time video stream |
| `/docs` | GET | OpenAPI Swagger docs |

---

## ⌨️ Keyboard Shortcuts

| Key | Action |
|---|---|
| `SPACE` | Toggle detection start/stop |
| `S` | Take snapshot |
| `ESC` | Close modal |

---

## 🛠️ Tuning the Detector

In `detector.py`, the `AntiSpoofDetector.analyze_face()` weights can be adjusted:

```python
real_confidence = (
    0.35 * t_norm +   # Texture entropy weight
    0.25 * s_norm +   # Sharpness weight
    0.20 * c_norm +   # Color variance weight
    0.20 * f_norm     # Frequency energy weight
)
```

The decision threshold is `0.48` — lower it to be more strict about flagging spoofs.

---

## 📦 Optional: DNN Face Detector

For more accurate face detection, download:
- [deploy.prototxt](https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt)
- [res10_300x300_ssd_iter_140000.caffemodel](https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel)

Place both files in `backend/models/`.

---

## 🔒 Privacy

All processing happens **locally on your machine**. No video data is sent to any external server.
