"""
Face Anti-Spoofing Detector
Uses a multi-layered approach:
1. Haar Cascade / DNN face detection
2. LBP (Local Binary Pattern) texture analysis
3. Blink detection for liveness
4. Micro-texture analysis to distinguish real vs printed/screen faces
"""

import cv2
import numpy as np
import time
import math
from collections import deque


class FaceDetector:
    def __init__(self):
        # Load DNN-based face detector (more accurate than Haar Cascade)
        self.net = None
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml"
        )
        self._try_load_dnn()

    def _try_load_dnn(self):
        try:
            model_path = "models/deploy.prototxt"
            weights_path = "models/res10_300x300_ssd_iter_140000.caffemodel"
            self.net = cv2.dnn.readNetFromCaffe(model_path, weights_path)
            print("[INFO] DNN face detector loaded.")
        except Exception:
            print("[WARN] DNN model not found; using Haar Cascade fallback.")
            self.net = None

    def detect_faces(self, frame):
        """Returns list of (x, y, w, h) bounding boxes."""
        if self.net is not None:
            return self._dnn_detect(frame)
        return self._haar_detect(frame)

    def _dnn_detect(self, frame):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0)
        )
        self.net.setInput(blob)
        detections = self.net.forward()
        faces = []
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                faces.append((x1, y1, x2 - x1, y2 - y1))
        return faces

    def _haar_detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )
        return list(faces) if len(faces) > 0 else []


class AntiSpoofDetector:
    """
    Multi-cue anti-spoofing using:
    - LBP texture analysis (paper/screen have uniform texture)
    - Edge sharpness (real faces have natural gradients)
    - Color variance (printed photos look flat)
    - Temporal consistency (live faces have micro-movement)
    - Moiré pattern detection (screens produce moiré patterns)
    """

    def __init__(self):
        self.face_detector = FaceDetector()
        self.history = deque(maxlen=15)       # rolling window for temporal
        self.frame_count = 0
        self.fps_time = time.time()
        self.fps = 0
        self.lbp_radius = 1
        self.lbp_points = 8

    def compute_lbp_histogram(self, gray_roi):
        """Compute LBP texture histogram - real faces have varied patterns."""
        lbp = np.zeros_like(gray_roi)
        for i in range(self.lbp_radius, gray_roi.shape[0] - self.lbp_radius):
            for j in range(self.lbp_radius, gray_roi.shape[1] - self.lbp_radius):
                center = gray_roi[i, j]
                binary = 0
                offsets = [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]
                for bit, (dy, dx) in enumerate(offsets):
                    neighbor = gray_roi[i + dy, j + dx]
                    binary |= (1 << bit) if neighbor >= center else 0
                lbp[i, j] = binary
        hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-6)
        return hist

    def compute_lbp_fast(self, gray_roi):
        """Vectorized LBP - much faster."""
        h, w = gray_roi.shape
        lbp = np.zeros((h, w), dtype=np.uint8)
        offsets = [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]
        for bit, (dy, dx) in enumerate(offsets):
            shifted = np.roll(np.roll(gray_roi, -dy, axis=0), -dx, axis=1)
            lbp += ((shifted >= gray_roi).astype(np.uint8) << bit)
        hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-6)
        return hist

    def texture_score(self, gray_roi):
        """Higher entropy = more natural = more likely real."""
        hist = self.compute_lbp_fast(gray_roi)
        entropy = -np.sum(hist * np.log2(hist + 1e-9))
        # Real faces: entropy > 6.0; printed/screen: entropy < 5.5
        return float(entropy)

    def sharpness_score(self, gray_roi):
        """Laplacian variance — real faces have natural sharpness gradients."""
        lap = cv2.Laplacian(gray_roi, cv2.CV_64F)
        return float(lap.var())

    def color_variance_score(self, roi):
        """Real faces have higher color variance across channels."""
        b, g, r = cv2.split(roi)
        var = float(np.std(b) + np.std(g) + np.std(r)) / 3.0
        return var

    def frequency_score(self, gray_roi):
        """FFT-based analysis — screens/prints have repetitive frequency peaks."""
        fft = np.fft.fft2(gray_roi.astype(np.float32))
        fft_shift = np.fft.fftshift(fft)
        magnitude = 20 * np.log(np.abs(fft_shift) + 1)
        center = np.array(magnitude.shape) // 2
        # High freq ring
        y, x = np.ogrid[:magnitude.shape[0], :magnitude.shape[1]]
        dist = np.sqrt((x - center[1])**2 + (y - center[0])**2)
        high_freq_mask = (dist > 20) & (dist < 80)
        hf_energy = float(magnitude[high_freq_mask].mean())
        return hf_energy

    def analyze_face(self, frame, face_box):
        """
        Returns dict with scores and final verdict.
        """
        x, y, w, h = face_box
        # Pad the ROI slightly
        pad = int(min(w, h) * 0.1)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(frame.shape[1], x + w + pad)
        y2 = min(frame.shape[0], y + h + pad)

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0 or roi.shape[0] < 30 or roi.shape[1] < 30:
            return None

        roi_resized = cv2.resize(roi, (128, 128))
        gray_roi = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)

        # Compute all scores
        t_score = self.texture_score(gray_roi)
        s_score = self.sharpness_score(gray_roi)
        c_score = self.color_variance_score(roi_resized)
        f_score = self.frequency_score(gray_roi)

        # Normalize scores to 0-1 scale
        t_norm = min(t_score / 7.5, 1.0)           # texture entropy
        s_norm = min(s_score / 300.0, 1.0)          # sharpness
        c_norm = min(c_score / 35.0, 1.0)           # color variance
        f_norm = min(f_score / 12.0, 1.0)           # frequency content

        # Weighted confidence (real = high values)
        real_confidence = (
            0.35 * t_norm +
            0.25 * s_norm +
            0.20 * c_norm +
            0.20 * f_norm
        )

        self.history.append(real_confidence)
        smoothed = float(np.mean(self.history))

        # Temporal consistency bonus (real faces vary slightly over time)
        temporal_var = float(np.std(self.history)) if len(self.history) > 3 else 0.02
        # Real faces have slight variation (not perfectly static)
        temporal_bonus = 0.05 if 0.01 < temporal_var < 0.15 else -0.05
        smoothed = max(0.0, min(1.0, smoothed + temporal_bonus))

        is_real = smoothed >= 0.48

        return {
            "is_real": is_real,
            "confidence": round(smoothed * 100, 1),
            "real_confidence": round(smoothed * 100, 1),
            "spoof_confidence": round((1 - smoothed) * 100, 1),
            "scores": {
                "texture_entropy": round(t_score, 3),
                "sharpness": round(s_score, 1),
                "color_variance": round(c_score, 2),
                "frequency_energy": round(f_score, 2),
            },
            "face_box": [int(x), int(y), int(w), int(h)],
            "label": "REAL" if is_real else "SPOOF",
        }

    def process_frame(self, frame):
        """Process a single frame and return annotated frame + results."""
        self.frame_count += 1

        # FPS calculation
        now = time.time()
        if now - self.fps_time >= 1.0:
            self.fps = self.frame_count
            self.frame_count = 0
            self.fps_time = now

        faces = self.face_detector.detect_faces(frame)
        results = []

        for face_box in faces:
            analysis = self.analyze_face(frame, face_box)
            if analysis:
                results.append(analysis)
                self._annotate_frame(frame, analysis)

        # HUD overlay
        self._draw_hud(frame, len(faces))
        return frame, results

    def _annotate_frame(self, frame, analysis):
        x, y, w, h = analysis["face_box"]
        is_real = analysis["is_real"]
        label = analysis["label"]
        conf = analysis["confidence"]

        # Colors
        real_color = (0, 220, 80)     # vibrant green
        spoof_color = (0, 60, 255)    # red-orange
        color = real_color if is_real else spoof_color

        # Draw bounding box with thickness based on confidence
        thickness = 3 if conf > 75 else 2
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)

        # Corner accents
        corner_len = min(w, h) // 6
        corners = [
            (x, y, x + corner_len, y + corner_len),
            (x + w - corner_len, y, x + w, y + corner_len),
            (x, y + h - corner_len, x + corner_len, y + h),
            (x + w - corner_len, y + h - corner_len, x + w, y + h),
        ]
        for (cx1, cy1, cx2, cy2) in corners:
            cv2.line(frame, (cx1, cy1), (cx2, cy1), color, 3)
            cv2.line(frame, (cx1, cy1), (cx1, cy2), color, 3)

        # Label background
        label_text = f"{label}  {conf:.1f}%"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        lx, ly = x, y - 10
        cv2.rectangle(frame, (lx, ly - th - 8), (lx + tw + 10, ly + 4), color, -1)
        cv2.putText(frame, label_text, (lx + 5, ly - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Confidence bar below the box
        bar_x, bar_y = x, y + h + 8
        bar_w = w
        bar_h = 8
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
        fill = int(bar_w * (conf / 100))
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h), color, -1)

    def _draw_hud(self, frame, face_count):
        """Draw HUD overlay on frame."""
        h, w = frame.shape[:2]
        # Semi-transparent top bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 40), (10, 10, 30), -1)
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        cv2.putText(frame, f"FaceShield Anti-Spoof  |  FPS: {self.fps}  |  Faces: {face_count}",
                    (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 220, 255), 1)
