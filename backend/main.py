"""
FaceShield - Face Anti-Spoofing Backend
FastAPI server with WebSocket streaming for real-time detection
"""

import cv2
import base64
import json
import asyncio
import time
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import os
import sys

# Add parent dir to path
sys.path.insert(0, os.path.dirname(__file__))
from detector import AntiSpoofDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("faceshield")

app = FastAPI(
    title="FaceShield Anti-Spoofing API",
    description="Real-time face liveness & spoof detection system",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount frontend static files (CSS, JS, images)
frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

# Global detector and camera state
detector = AntiSpoofDetector()
camera_active = False
camera = None
detection_log = []
session_stats = {
    "total_detections": 0,
    "real_count": 0,
    "spoof_count": 0,
    "session_start": time.time(),
}


def get_camera():
    global camera
    if camera is None or not camera.isOpened():
        for idx in [0, 1, 2]:
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                camera = cap
                logger.info(f"[Camera] Opened camera index {idx}")
                return camera
        logger.error("[Camera] No camera found!")
        return None
    return camera


@app.get("/")
async def root():
    """Serve the frontend HTML."""
    index_path = os.path.join(frontend_dir, "index.html")
    if os.path.exists(index_path):
        # IMPORTANT: specify utf-8 to handle emoji/special chars on Windows
        with open(index_path, "r", encoding="utf-8") as f:
            content = f.read()
        return HTMLResponse(content, media_type="text/html; charset=utf-8")
    return JSONResponse({"message": "FaceShield API running. Frontend not found."})


@app.get("/api/status")
async def get_status():
    return {
        "status": "running",
        "camera_active": camera_active,
        "session_stats": session_stats,
        "uptime": round(time.time() - session_stats["session_start"], 1),
    }


@app.post("/api/camera/start")
async def start_camera():
    global camera_active
    cam = get_camera()
    if cam is None:
        return JSONResponse({"success": False, "error": "No camera available"}, status_code=500)
    camera_active = True
    return {"success": True, "message": "Camera started"}


@app.post("/api/camera/stop")
async def stop_camera():
    global camera_active, camera
    camera_active = False
    if camera:
        camera.release()
        camera = None
    return {"success": True, "message": "Camera stopped"}


@app.get("/api/log")
async def get_detection_log():
    return {"log": detection_log[-50:]}  # Last 50 detections


@app.post("/api/analyze")
async def analyze_image(data: dict):
    """Analyze a base64-encoded image frame."""
    try:
        img_data = data.get("image", "")
        if "," in img_data:
            img_data = img_data.split(",")[1]
        img_bytes = base64.b64decode(img_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return JSONResponse({"error": "Invalid image"}, status_code=400)

        annotated, results = detector.process_frame(frame)

        # Update stats
        for r in results:
            session_stats["total_detections"] += 1
            if r["is_real"]:
                session_stats["real_count"] += 1
            else:
                session_stats["spoof_count"] += 1
            detection_log.append({
                "timestamp": time.time(),
                "label": r["label"],
                "confidence": r["confidence"],
                "is_real": r["is_real"],
            })

        # Encode annotated frame back to base64
        _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_b64 = base64.b64encode(buffer).decode("utf-8")

        return {
            "frame": f"data:image/jpeg;base64,{frame_b64}",
            "results": results,
            "face_count": len(results),
        }
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.websocket("/ws/detect")
async def websocket_detect(websocket: WebSocket):
    """WebSocket endpoint for real-time camera streaming."""
    await websocket.accept()
    global camera_active, camera, session_stats
    logger.info("[WS] Client connected")

    cam = get_camera()
    if cam is None:
        await websocket.send_json({"error": "No camera available"})
        await websocket.close()
        return

    camera_active = True
    frame_skip = 0

    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                logger.warning("[WS] Failed to read frame")
                await asyncio.sleep(0.05)
                continue

            frame_skip += 1
            if frame_skip % 2 != 0:  # Process every other frame for performance
                continue

            # Process frame
            annotated, results = detector.process_frame(frame)

            # Update stats
            for r in results:
                session_stats["total_detections"] += 1
                if r["is_real"]:
                    session_stats["real_count"] += 1
                else:
                    session_stats["spoof_count"] += 1

            # Encode frame
            _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame_b64 = base64.b64encode(buffer).decode("utf-8")

            payload = {
                "frame": f"data:image/jpeg;base64,{frame_b64}",
                "results": results,
                "face_count": len(results),
                "stats": session_stats,
            }

            await websocket.send_json(payload)
            await asyncio.sleep(0.03)  # ~30 FPS target

    except WebSocketDisconnect:
        logger.info("[WS] Client disconnected")
    except Exception as e:
        logger.error(f"[WS] Error: {e}")
    finally:
        camera_active = False


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  FaceShield Anti-Spoofing Detection System")
    print("  Dashboard: http://localhost:8000")
    print("  API Docs:  http://localhost:8000/docs")
    print("=" * 60 + "\n")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
