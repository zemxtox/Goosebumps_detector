#!/usr/bin/env python3
"""
CHILLER Server - Color Display with Grayscale Detection
Detection uses grayscale (research-validated)
Dashboard shows color (better visual feedback)
"""
import base64, os, time, pathlib, traceback, threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Tuple
from concurrent.futures import ThreadPoolExecutor
import logging

import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, redirect
from flask_socketio import SocketIO

# Configure logging for better debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------- Config --------------------
import os

# Use environment variables with defaults for cloud deployment
HOST = os.environ.get("HOST", "0.0.0.0")  # Railway requires 0.0.0.0
PORT = int(os.environ.get("PORT", 8000))  # Railway will set this automatically

# Detection configuration with environment variable support
BASELINE_FRAMES = int(os.environ.get("BASELINE_FRAMES", 5))  # Reduced for faster baseline
DETECTION_THRESHOLD = float(os.environ.get("DETECTION_THRESHOLD", 25.0))  # Slightly lower threshold
VIDEO_DETECTION_THRESHOLD = float(os.environ.get("VIDEO_DETECTION_THRESHOLD", 60.0))  # Higher threshold for uploaded videos
SAMPLING_RATE = int(os.environ.get("SAMPLING_RATE", 2))

# Performance optimization settings
MAX_IMAGE_SIZE = 800  # Increased for better quality
JPEG_QUALITY = 90  # Better quality for improved detection
PROCESSING_THREADS = 4  # Number of processing threads
FRAME_SKIP_THRESHOLD = 100  # Skip frames if processing is too slow

FREQ_MIN_MM = 0.23
FREQ_MAX_MM = 0.75

ROI_WIDTH = 160
ROI_HEIGHT = 120
CLAHE_CLIP = 2.0

LED_OFF_LEVEL = 0
LED_MIN_LEVEL = 30
LED_MAX_LEVEL = 100

# Use environment variable to control detection saving
SAVE_DETECTIONS = os.environ.get("SAVE_DETECTIONS", "True").lower() in ["true", "1", "yes"]
DETECTION_DIR = os.environ.get("DETECTION_DIR", "chiller_detections")

# Create detection directory if saving is enabled
if SAVE_DETECTIONS:
    pathlib.Path(DETECTION_DIR).mkdir(exist_ok=True)

# -------------------- Flask --------------------
app = Flask(__name__, template_folder=".", static_folder=".", static_url_path="")

# Configure for production deployment
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

# Configure CORS for Railway deployment
allowed_origins = []
if os.environ.get('RAILWAY_ENVIRONMENT'):
    # In production, allow Railway domain
    railway_domain = os.environ.get('RAILWAY_STATIC_URL', '*')
    allowed_origins = [railway_domain, '*']
else:
    # In development, allow all origins
    allowed_origins = '*'

socketio = SocketIO(app, cors_allowed_origins=allowed_origins, async_mode="threading")


# -------------------- Performance Monitoring --------------------
@dataclass
class PerformanceMetrics:
    processing_times: list = field(default_factory=list)
    frame_rates: list = field(default_factory=list)
    last_frame_time: float = 0.0
    avg_processing_time: float = 0.0
    current_fps: float = 0.0
    
    def update(self, processing_time: float):
        now = time.time()
        self.processing_times.append(processing_time)
        
        # Calculate FPS
        if self.last_frame_time > 0:
            frame_delta = now - self.last_frame_time
            fps = 1.0 / frame_delta if frame_delta > 0 else 0
            self.frame_rates.append(fps)
            self.current_fps = fps
        
        self.last_frame_time = now
        
        # Keep only last 30 measurements
        if len(self.processing_times) > 30:
            self.processing_times.pop(0)
        if len(self.frame_rates) > 30:
            self.frame_rates.pop(0)
            
        # Update averages
        self.avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0

# -------------------- Device State --------------------
@dataclass
class ChillerState:
    device_id: str
    baseline_frames: list = field(default_factory=list)
    baseline_power: float = 0.0
    baseline_established: bool = False
    current_intensity: float = 0.0
    led_level: int = 0
    frame_count: int = 0
    detection_count: int = 0
    max_intensity_seen: float = 0.0
    last_seen_ts: float = 0.0
    roi_x: int = 0
    roi_y: int = 0
    last_processed_frame: int = 0
    skip_frames: int = 1  # Reduced for better responsiveness
    performance: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    processing_queue_size: int = 0

device_states: Dict[str, ChillerState] = {}

# Inactive device cleanup threshold (60 seconds)
INACTIVE_DEVICE_THRESHOLD = 60

# Thread pool for processing frames
executor = ThreadPoolExecutor(max_workers=PROCESSING_THREADS)

# Performance monitoring
performance_stats = {
    "total_frames_processed": 0,
    "total_detections": 0,
    "avg_processing_time": 0.0,
    "peak_processing_time": 0.0,
    "active_devices": 0
}

def get_state(device_id: str) -> ChillerState:
    if device_id not in device_states:
        device_states[device_id] = ChillerState(device_id=device_id)
        device_list = [{"id": did, "baseline_ready": st.baseline_established} 
                      for did, st in device_states.items()]
        socketio.emit("devices", {"devices": device_list})
    
    # Update last seen timestamp
    device_states[device_id].last_seen_ts = time.time()
    return device_states[device_id]

def cleanup_inactive_devices():
    """Remove devices that haven't sent frames in a while"""
    now = time.time()
    inactive_devices = []
    
    for device_id, state in device_states.items():
        if now - state.last_seen_ts > INACTIVE_DEVICE_THRESHOLD:
            inactive_devices.append(device_id)
    
    for device_id in inactive_devices:
        print(f"[CLEANUP] Removing inactive device: {device_id}")
        del device_states[device_id]
    
    if inactive_devices:
        # Notify clients about updated device list
        device_list = [{"id": did, "baseline_ready": st.baseline_established} 
                      for did, st in device_states.items()]
        socketio.emit("devices", {"devices": device_list})
        
    # Schedule next cleanup
    socketio.sleep(30)  # Check every 30 seconds

# -------------------- CHILLER Algorithm (GRAYSCALE ONLY) --------------------
def angular_averaging(fft_2d: np.ndarray) -> np.ndarray:
    """Convert 2D DFT to 1D spectrum via angular averaging."""
    center = np.array(fft_2d.shape) // 2
    y, x = np.indices(fft_2d.shape)
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2).astype(int)
    
    max_r = min(center)
    radial_profile = np.zeros(max_r)
    
    for radius in range(max_r):
        mask = (r == radius)
        if mask.sum() > 0:
            radial_profile[radius] = fft_2d[mask].mean()
    
    return radial_profile

def calculate_goosebump_power(gray_roi: np.ndarray, pixel_size_mm: float = 0.25) -> Tuple[float, dict]:
    """
    CHILLER goosebump detection algorithm - GRAYSCALE ONLY.
    This MUST remain grayscale for research validity.
    """
    h, w = gray_roi.shape
    
    if h < 32 or w < 32:
        return 0.0, {"error": "roi_too_small"}
    
    if h != ROI_HEIGHT or w != ROI_WIDTH:
        gray_roi = cv2.resize(gray_roi, (ROI_WIDTH, ROI_HEIGHT))
    
    roi_mean = gray_roi.mean()
    roi_std = gray_roi.std()
    
    if roi_std < 1.0:
        return 0.0, {"error": "low_contrast"}
    
    gray_norm = (gray_roi - roi_mean) / roi_std
    
    # 2D FFT (grayscale only!)
    fft_2d = np.fft.fft2(gray_norm)
    fft_2d_shifted = np.fft.fftshift(fft_2d)
    power_spectrum_2d = np.abs(fft_2d_shifted) ** 2
    
    # Angular averaging
    radial_power = angular_averaging(power_spectrum_2d)
    
    # Frequency conversion
    nyquist_freq = 0.5
    freq_axis = np.linspace(0, nyquist_freq, len(radial_power))
    freq_min_pixels = FREQ_MIN_MM * pixel_size_mm
    freq_max_pixels = FREQ_MAX_MM * pixel_size_mm
    
    mask = (freq_axis >= freq_min_pixels) & (freq_axis <= freq_max_pixels)
    
    if mask.sum() > 0:
        max_power = radial_power[mask].max()
        mean_power = radial_power[mask].mean()
    else:
        max_power = 0.0
        mean_power = 0.0
    
    metrics = {
        "max_power": float(max_power),
        "mean_power": float(mean_power),
        "roi_std": float(roi_std)
    }
    
    return max_power, metrics

# -------------------- Image Processing --------------------
def enhance_skin_image(gray: np.ndarray) -> np.ndarray:
    """CLAHE enhancement - grayscale only."""
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=(8, 8))
    return clahe.apply(gray)

def auto_select_roi(gray: np.ndarray, prev_x: int = 0, prev_y: int = 0) -> Tuple[int, int]:
    """Select ROI (center-biased) - uses grayscale for consistency."""
    h, w = gray.shape
    x = (w - ROI_WIDTH) // 2
    y = (h - ROI_HEIGHT) // 2
    
    if prev_x > 0 and prev_y > 0:
        if prev_x + ROI_WIDTH <= w and prev_y + ROI_HEIGHT <= h:
            return prev_x, prev_y
    
    x = max(0, min(x, w - ROI_WIDTH))
    y = max(0, min(y, h - ROI_HEIGHT))
    return x, y

# -------------------- Routes --------------------
@app.route("/")
def index():
    return render_template("chiller_dashboard.html")

@app.route("/health")
def health_check():
    """Health check endpoint for Railway"""
    return jsonify({
        "status": "healthy",
        "service": "CHILLER Goosebump Detector",
        "version": "1.0.0",
        "timestamp": time.time(),
        "active_devices": len(device_states)
    })

@app.route("/manifest.json")
def manifest():
    return app.send_static_file("manifest.json")

@app.route("/service-worker.js")
def service_worker():
    response = app.send_static_file("service-worker.js")
    response.headers['Content-Type'] = 'application/javascript'
    response.headers['Service-Worker-Allowed'] = '/'
    return response

@app.route("/icons/<path:filename>")
def icons(filename):
    return app.send_static_file(f"icons/{filename}")

# Add security headers for PWA
@app.after_request
def add_security_headers(response):
    # Add PWA-specific headers for service worker
    if request.endpoint == 'service_worker':
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
    
    return response

@app.route("/devices", methods=["GET"])
def list_devices():
    devices_info = []
    for device_id, state in device_states.items():
        devices_info.append({
            "id": device_id,
            "baseline_ready": state.baseline_established,
            "detections": state.detection_count,
            "max_intensity": state.max_intensity_seen,
            "performance": {
                "fps": round(state.performance.current_fps, 1),
                "avg_processing_time": round(state.performance.avg_processing_time * 1000, 1),
                "queue_size": state.processing_queue_size
            }
        })
    return jsonify({"devices": devices_info})

@app.route("/performance", methods=["GET"])
def get_performance_stats():
    """Get system performance statistics."""
    performance_stats["active_devices"] = len(device_states)
    
    # Calculate system-wide metrics
    total_fps = sum(state.performance.current_fps for state in device_states.values())
    avg_system_fps = total_fps / len(device_states) if device_states else 0
    
    return jsonify({
        **performance_stats,
        "avg_system_fps": round(avg_system_fps, 1),
        "memory_usage": {
            "device_states": len(device_states),
            "total_baseline_frames": sum(len(state.baseline_frames) for state in device_states.values())
        },
        "processing_load": {
            "total_queue_size": sum(state.processing_queue_size for state in device_states.values()),
            "thread_pool_active": executor._threads and len(executor._threads) or 0
        }
    })

@app.route("/pwa-debug")
def pwa_debug():
    """Debug endpoint to check PWA setup"""
    import os
    debug_info = {
        "manifest_exists": os.path.exists("manifest.json"),
        "service_worker_exists": os.path.exists("service-worker.js"),
        "icons_dir_exists": os.path.exists("icons"),
        "icon_192_exists": os.path.exists("icons/icon-192x192.png"),
        "icon_512_exists": os.path.exists("icons/icon-512x512.png"),
        "is_https": request.is_secure,
        "user_agent": request.headers.get('User-Agent', ''),
        "host": request.host
    }
    return jsonify(debug_info)

@app.route("/reset_baseline/<device_id>", methods=["POST"])
def reset_baseline(device_id):
    """Reset baseline for a device."""
    if device_id in device_states:
        st = device_states[device_id]
        st.baseline_frames.clear()
        st.baseline_power = 0.0
        st.baseline_established = False
        st.current_intensity = 0.0
        st.led_level = 0
        print(f"[RESET] Baseline reset for {device_id}")
        return jsonify({"status": "baseline_reset", "device_id": device_id})
    return jsonify({"error": "device_not_found"}), 404

def process_frame_async(device_id: str, jpg_b64: str, frame_number: int = None):
    """Asynchronous frame processing function."""
    start_time = time.time()
    
    try:
        st = get_state(device_id)
        
        # Update performance stats
        performance_stats["total_frames_processed"] += 1
        
        # If frame_number is provided, use it instead of incrementing
        if frame_number is not None:
            st.frame_count = frame_number
        else:
            st.frame_count += 1

        # Adaptive frame skipping based on processing load
        is_camera_feed = device_id.startswith('mobile-camera-') or device_id.startswith('camera-')
        if is_camera_feed and st.baseline_established and st.processing_queue_size > 2:
            # Skip frames if processing queue is too large
            frames_since_last = st.frame_count - st.last_processed_frame
            if frames_since_last < st.skip_frames:
                return {
                    "status": "skipped",
                    "device_id": device_id,
                    "frame": st.frame_count,
                    "baseline_ready": st.baseline_established,
                    "intensity": round(st.current_intensity, 1),
                    "led_level": st.led_level,
                    "detect": False
                }
        
        st.last_processed_frame = st.frame_count

        # Decode image with error handling
        try:
            jpg_bytes = base64.b64decode(jpg_b64)
            arr = np.frombuffer(jpg_bytes, np.uint8)
            img_color = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception as e:
            logger.error(f"Image decode error for {device_id}: {e}")
            return {"error": "invalid_image_data"}
        
        if img_color is None:
            return {"error": "invalid_image"}

        # Intelligent image resizing
        h, w = img_color.shape[:2]
        if max(h, w) > MAX_IMAGE_SIZE:
            scale = MAX_IMAGE_SIZE / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            img_color = cv2.resize(img_color, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Convert to grayscale FOR DETECTION ONLY
        gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        gray_enhanced = enhance_skin_image(gray)

        # Select and extract ROI (grayscale)
        st.roi_x, st.roi_y = auto_select_roi(gray_enhanced, st.roi_x, st.roi_y)
        gray_roi = gray_enhanced[st.roi_y:st.roi_y+ROI_HEIGHT, 
                                 st.roi_x:st.roi_x+ROI_WIDTH]

        # Calculate goosebump power (GRAYSCALE DETECTION)
        max_power, metrics = calculate_goosebump_power(gray_roi)

        # === BASELINE COLLECTION ===
        status = ""
        if not st.baseline_established:
            st.baseline_frames.append(float(max_power))
            
            if len(st.baseline_frames) >= BASELINE_FRAMES:
                st.baseline_power = float(np.mean(st.baseline_frames))
                st.baseline_established = True
                logger.info(f"[{device_id}] ✓ Baseline established: {st.baseline_power:.4f}")
                status = "baseline_complete"
                st.led_level = 0
            else:
                status = f"baseline_{len(st.baseline_frames)}/{BASELINE_FRAMES}"
                st.led_level = -1
                st.current_intensity = 0.0
        else:
            if st.baseline_power > 0:
                st.current_intensity = ((max_power - st.baseline_power) / st.baseline_power) * 100.0
            else:
                st.current_intensity = 0.0
            
            st.max_intensity_seen = max(st.max_intensity_seen, st.current_intensity)

        # === DETECTION LOGIC ===
        detect = False
        
        # Use different threshold for uploaded videos vs. live camera
        threshold = VIDEO_DETECTION_THRESHOLD if device_id.startswith('video-upload-') else DETECTION_THRESHOLD
        
        if st.baseline_established and st.current_intensity >= threshold:
            detect = True
            st.detection_count += 1
            performance_stats["total_detections"] += 1
            
            led_intensity = st.current_intensity - threshold
            st.led_level = int(np.clip(
                LED_MIN_LEVEL + (led_intensity / 50.0) * (LED_MAX_LEVEL - LED_MIN_LEVEL),
                LED_MIN_LEVEL,
                LED_MAX_LEVEL
            ))
            
            status = f"GOOSEBUMP_DETECTED"
            
            # Async save detection
            if SAVE_DETECTIONS:
                try:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    viz = img_color.copy()
                    cv2.rectangle(viz, (st.roi_x, st.roi_y),
                                (st.roi_x + ROI_WIDTH, st.roi_y + ROI_HEIGHT),
                                (0, 255, 0), 2)
                    text = f"GOOSEBUMPS: {st.current_intensity:.1f}%"
                    cv2.putText(viz, text, (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                    
                    fname = f"{device_id}_gb_{ts}_i{st.current_intensity:.0f}.jpg"
                    # Save asynchronously
                    executor.submit(cv2.imwrite, os.path.join(DETECTION_DIR, fname), viz)
                except Exception as e:
                    logger.warning(f"Save error for {device_id}: {e}")
        else:
            if st.baseline_established:
                status = "monitoring"
                st.led_level = 0

        # === CREATE COLOR VISUALIZATION FOR DASHBOARD ===
        viz_color = img_color.copy()
        
        # Draw ROI on COLOR image
        roi_color = (0, 255, 0) if detect else (128, 128, 128)
        cv2.rectangle(viz_color, (st.roi_x, st.roi_y),
                     (st.roi_x + ROI_WIDTH, st.roi_y + ROI_HEIGHT),
                     roi_color, 2)
        
        # Status overlay with better visibility
        if not st.baseline_established:
            color = (0, 0, 255)
            text = f"BASELINE: {len(st.baseline_frames)}/{BASELINE_FRAMES}"
        elif detect:
            color = (0, 255, 0)
            text = f"GOOSEBUMPS: {st.current_intensity:.1f}%"
        else:
            color = (200, 200, 200)
            text = f"Monitoring: {st.current_intensity:.1f}%"
        
        # Add background for better text visibility
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        cv2.rectangle(viz_color, (5, 5), (text_size[0] + 15, text_size[1] + 15), (0, 0, 0), -1)
        cv2.putText(viz_color, text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Encode COLOR output image for dashboard with optimized quality
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
        ok, jpeg = cv2.imencode(".jpg", viz_color, encode_params)
        out_b64 = base64.b64encode(jpeg.tobytes()).decode("utf-8") if ok else jpg_b64

        # Update performance metrics
        processing_time = time.time() - start_time
        st.performance.update(processing_time)
        
        # Update global performance stats
        performance_stats["avg_processing_time"] = (
            performance_stats["avg_processing_time"] * 0.9 + processing_time * 0.1
        )
        performance_stats["peak_processing_time"] = max(
            performance_stats["peak_processing_time"], processing_time
        )

        # === PREPARE RESPONSE PACKET ===
        packet = {
            "device_id": device_id,
            "img": out_b64,
            "baseline_established": st.baseline_established,
            "baseline_power": round(st.baseline_power, 4),
            "current_power": round(max_power, 4),
            "goosebump_intensity": round(st.current_intensity, 1),
            "led_level": st.led_level,
            "detect": detect,
            "frame_number": st.frame_count,
            "detection_count": st.detection_count,
            "timestamp": datetime.now().isoformat(),
            "status": status,
            "max_power": round(metrics.get("max_power", 0), 4),
            "mean_power": round(metrics.get("mean_power", 0), 4),
            "roi_std": round(metrics.get("roi_std", 0), 2),
            "performance": {
                "processing_time": round(processing_time * 1000, 1),  # ms
                "fps": round(st.performance.current_fps, 1),
                "avg_processing_time": round(st.performance.avg_processing_time * 1000, 1)
            }
        }

        return packet

    except Exception as e:
        logger.error(f"Frame processing error for {device_id}: {e}")
        traceback.print_exc()
        return {"error": str(e)}

@app.route("/upload_frame", methods=["POST"])
def upload_frame():
    """Main frame processing endpoint with async processing."""
    try:
        data = request.get_json(silent=True) or {}
        device_id = (data.get("device_id") or request.remote_addr or "unknown").strip()
        jpg_b64 = data.get("jpg_b64")
        frame_number = data.get("frame_number")
        
        if not jpg_b64:
            return jsonify({"error": "missing_jpg_b64"}), 400

        st = get_state(device_id)
        st.processing_queue_size += 1
        
        try:
            # Process frame asynchronously for better performance
            future = executor.submit(process_frame_async, device_id, jpg_b64, frame_number)
            result = future.result(timeout=5.0)  # 5 second timeout
            
            if "error" in result:
                return jsonify(result), 400
            
            # Emit to socket clients
            socketio.emit("frame", result)
            
            return jsonify({
                "status": "ok",
                "device_id": device_id,
                "frame": result.get("frame_number", st.frame_count),
                "baseline_ready": result.get("baseline_established", False),
                "intensity": result.get("goosebump_intensity", 0),
                "led_level": result.get("led_level", 0),
                "detect": result.get("detect", False),
                "performance": result.get("performance", {})
            }), 200
            
        finally:
            st.processing_queue_size = max(0, st.processing_queue_size - 1)

    except Exception as e:
        logger.error(f"Upload frame error: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# -------------------- Background Tasks --------------------
@socketio.on('connect')
def handle_connect():
    print(f"[INFO] Client connected")
    # Send current device list to new client
    device_list = [{"id": did, "baseline_ready": st.baseline_established} 
                  for did, st in device_states.items()]
    socketio.emit("devices", {"devices": device_list})

def start_background_tasks():
    """Start background tasks like device cleanup"""
    print("[INFO] Starting background tasks")
    socketio.start_background_task(background_cleanup_task)

def background_cleanup_task():
    """Background task to periodically clean up inactive devices"""
    print("[INFO] Device cleanup task started")
    while True:
        cleanup_inactive_devices()

# -------------------- Main --------------------
if __name__ == "__main__":
    print("="*80)
    print(" CHILLER - Goosebump Detection System")
    print(" Detection: GRAYSCALE (research-validated)")
    print(" Display: COLOR (enhanced visual feedback)")
    print("="*80)
    
    # Check if running on Railway
    railway_url = os.environ.get('RAILWAY_STATIC_URL')
    if railway_url:
        print(f" 🚀 Railway Deployment: {railway_url}")
        print(f" 📱 PWA Installation: Available (HTTPS)")
        print(f" 📷 Camera Access: Available (HTTPS)")
    else:
        print(f" 🏠 Local Development: http://{HOST}:{PORT}")
        if HOST == "localhost" or HOST == "127.0.0.1":
            print(f" 📱 PWA Installation: Available")
            print(f" 📷 Camera Access: Available")
        else:
            print(f" 📱 PWA Installation: Requires HTTPS")
            print(f" 📷 Camera Access: Requires HTTPS")
    
    print(f" ⚙️ Sampling Rate: {SAMPLING_RATE} Hz")
    print(f" 🎯 Detection Threshold: {DETECTION_THRESHOLD}% change from baseline")
    print(f" 📹 Video Detection Threshold: {VIDEO_DETECTION_THRESHOLD}% change from baseline")
    print(f" 🧹 Inactive device cleanup: {INACTIVE_DEVICE_THRESHOLD} seconds")
    print(f" 💾 Save Detections: {SAVE_DETECTIONS}")
    print("="*80)
    
    # Start background tasks
    start_background_tasks()
    
    # Run the server
    # Check if we're in production (Railway) or development
    railway_url = os.environ.get('RAILWAY_STATIC_URL')
    is_production = railway_url is not None or os.environ.get('RAILWAY_ENVIRONMENT') is not None
    
    if is_production:
        # Production deployment - use proper ASGI/WSGI server
        print(f" 🚀 Starting production server on {HOST}:{PORT}")
        
        # Try different server backends
        server_started = False
        
        # Option 1: Try eventlet
        try:
            import eventlet
            print(" 🔧 Using Eventlet server (recommended for SocketIO)")
            socketio.run(app, host=HOST, port=PORT, debug=False, 
                        server='eventlet', use_reloader=False, log_output=True)
            server_started = True
        except ImportError:
            print(" ⚠️ Eventlet not available")
        except Exception as e:
            print(f" ⚠️ Eventlet failed: {e}")
        
        # Option 2: Try gevent if eventlet failed
        if not server_started:
            try:
                import gevent
                print(" 🔧 Using Gevent server")
                socketio.run(app, host=HOST, port=PORT, debug=False, 
                            server='gevent', use_reloader=False, log_output=True)
                server_started = True
            except ImportError:
                print(" ⚠️ Gevent not available")
            except Exception as e:
                print(f" ⚠️ Gevent failed: {e}")
        
        # Option 3: Last resort - threading with unsafe werkzeug
        if not server_started:
            print(" ⚠️ Using threading server (not optimal for production)")
            try:
                socketio.run(app, host=HOST, port=PORT, debug=False, 
                            allow_unsafe_werkzeug=True, use_reloader=False, 
                            log_output=True)
            except Exception as e:
                print(f" ❌ All server options failed: {e}")
                print(" 💡 Try running: pip install eventlet gevent")
                raise
    else:
        # Development server
        print(f" 🏠 Starting development server on {HOST}:{PORT}")
        socketio.run(app, host=HOST, port=PORT, debug=True)

