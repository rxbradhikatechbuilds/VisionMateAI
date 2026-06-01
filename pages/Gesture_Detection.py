import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import mediapipe as mp
import time
import sqlite3
import streamlit as st
import pandas as pd
from datetime import datetime
from queue import Queue

# Import the new universal camera utilities
import camera_utils as cam

# ============================================
# PAGE CONFIG & ATTRACTIVE DARK THEME
# ============================================
st.set_page_config(page_title="Gesture Detection", page_icon="✋", layout="wide")

st.markdown("""
<style>
    /* Base background - soft dark blue/gray */
    .stApp {
        background: linear-gradient(135deg, #1a1c2e 0%, #1e2035 100%);
    }
    /* Sidebar */
    .css-1d391kg, .css-12oz5g7 {
        background-color: #151728;
        border-right: 1px solid #2c2f42;
    }
    /* All text */
    h1, h2, h3, .stMarkdown, label, .stMetricLabel {
        color: #eef2ff !important;
    }
    /* Main title */
    .main-title {
        text-align: center;
        margin-bottom: 1rem;
    }
    /* Gesture card */
    .gesture-card {
        background: #23263b;
        border-radius: 24px;
        padding: 1.5rem;
        box-shadow: 0 8px 20px rgba(0,0,0,0.3);
        border: 1px solid #ff4444;
        text-align: center;
        margin-bottom: 1rem;
    }
    .gesture-name {
        font-size: 2.5rem;
        font-weight: bold;
        color: #ff6b6b;
        margin: 0.5rem 0;
    }
    .confidence-text {
        font-size: 1.2rem;
        color: #cbd5ff;
    }
    /* Progress bar */
    .progress-bar {
        width: 100%;
        background-color: #2d3047;
        border-radius: 10px;
        overflow: hidden;
        margin: 10px 0;
    }
    .progress-fill {
        height: 20px;
        background-color: #ff4444;
        width: 0%;
        border-radius: 10px;
        transition: width 0.3s ease;
    }
    /* Metric boxes */
    .metric-box {
        background: #23263b;
        border-radius: 16px;
        padding: 0.8rem;
        text-align: center;
        border-left: 4px solid #ff4444;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    /* Camera container */
    .camera-container {
        background-color: #0f111d;
        border-radius: 24px;
        padding: 10px;
        border: 2px solid #ff4444;
        box-shadow: 0 8px 20px rgba(0,0,0,0.4);
    }
    /* Buttons */
    .stButton > button {
        background-color: #ff4444;
        color: white;
        border-radius: 12px;
        border: none;
        font-weight: 600;
        transition: 0.2s;
    }
    .stButton > button:hover {
        background-color: #ff2222;
        transform: scale(1.02);
    }
    /* Dataframe */
    .dataframe {
        background-color: #1e2137 !important;
        color: #eef2ff !important;
    }
    /* Sidebar text */
    .sidebar .sidebar-content {
        color: #eef2ff;
    }
    /* Info/Warning boxes */
    .stAlert {
        background-color: #23263b !important;
        color: #eef2ff !important;
        border-left: 4px solid #ff4444 !important;
    }
    /* Selectbox, slider etc */
    .stSelectbox label, .stSlider label {
        color: #eef2ff !important;
    }
    /* Metric value */
    [data-testid="stMetricValue"] {
        color: #ff6b6b !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# DATABASE SETUP
# ============================================
DB_FILE = "gesture_history.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    conn.execute('''CREATE TABLE IF NOT EXISTS gesture_logs
        (id INTEGER PRIMARY KEY, gesture_name TEXT, confidence_percentage TEXT,
         time_of_detection TEXT, date_of_detection TEXT)''')
    conn.close()

def save_gesture(name, conf):
    now = datetime.now()
    conn = sqlite3.connect(DB_FILE)
    conn.execute("INSERT INTO gesture_logs (gesture_name, confidence_percentage, time_of_detection, date_of_detection) VALUES (?,?,?,?)",
                 (name, f"{conf}%", now.strftime("%H:%M:%S"), now.strftime("%Y-%m-%d")))
    conn.commit()
    conn.close()

def fetch_logs():
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT * FROM gesture_logs ORDER BY id DESC LIMIT 50", conn)
    conn.close()
    return df

init_db()

# ============================================
# MEDIAPIPE SETUP (shared)
# ============================================
model_path = 'D:/Github/Vision_mate_AI/pages/gesture_recognizer.task'   # ⬅️ Change to your actual path

if not os.path.exists(model_path):
    st.error("Model file 'gesture_recognizer.task' not found. Please update the path.")
    st.stop()

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

detection_queue = Queue(maxsize=1)

def callback(result, output_image, timestamp_ms):
    if result.gestures and len(result.gestures) > 0:
        top = result.gestures[0][0]
        gesture_name = top.category_name
        confidence = int(top.score * 100)
        if gesture_name != "None":
            try:
                detection_queue.get_nowait()
            except:
                pass
            detection_queue.put((gesture_name, confidence))

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=1,
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3,
    min_tracking_confidence=0.3,
    result_callback=callback
)

# ============================================
# UI HEADER
# ============================================
st.title("✋ Real-Time Gesture Recognition")
st.markdown("Detects: **thumbs_up** · **victory** · **open_palm** · **fist** · **pointing**")

# ============================================
# SIDEBAR SETTINGS (shared)
# ============================================
resolution = st.sidebar.selectbox(
    "Resolution (lower = faster)",
    ["Very Low (160x120)", "Low (320x240)", "Medium (480x360)"]
)
if resolution == "Very Low (160x120)":
    width, height = 160, 120
elif resolution == "Low (320x240)":
    width, height = 320, 240
else:
    width, height = 480, 360

process_every_n = st.sidebar.slider("Process every N frames (higher = faster)", 1, 15, 5)
run_stream = st.checkbox("🔄 START CAMERA", value=False)

# Layout
col_left, col_right = st.columns([3, 2])

with col_left:
    st.markdown("<div class='camera-container'>", unsafe_allow_html=True)
    frame_placeholder = st.empty()
    st.markdown("</div>", unsafe_allow_html=True)

with col_right:
    gesture_display = st.empty()
    progress_placeholder = st.empty()
    st.markdown("### 📊 Live Stats")
    metric_col1, metric_col2 = st.columns(2)
    with metric_col1:
        st.markdown("<div class='metric-box'>", unsafe_allow_html=True)
        total_detections = st.empty()
        st.markdown("</div>", unsafe_allow_html=True)
    with metric_col2:
        st.markdown("<div class='metric-box'>", unsafe_allow_html=True)
        last_gesture_time = st.empty()
        st.markdown("</div>", unsafe_allow_html=True)

history_placeholder = st.empty()

# ============================================
# DETECTION LOOP (works both locally and on cloud)
# ============================================
def process_frame_and_update(frame, recognizer, frame_counter, last_saved, total_detected, current_gesture):
    """
    Process a single frame: run MediaPipe (with interval), update UI, return annotated frame.
    """
    # MediaPipe async call (frame skipping)
    if frame_counter % process_every_n == 0:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        recognizer.recognize_async(mp_image, frame_counter)

    # Get latest detection from queue
    try:
        gesture_name, confidence = detection_queue.get_nowait()
        current_gesture = {"name": gesture_name, "confidence": confidence}
    except:
        pass

    gesture = current_gesture["name"]
    confidence = current_gesture["confidence"]

    # Save to DB if new gesture
    if gesture != "None" and gesture != last_saved:
        save_gesture(gesture, confidence)
        last_saved = gesture
        total_detected += 1
        history_placeholder.dataframe(fetch_logs(), use_container_width=True, hide_index=True)

    # Update right panel UI
    icon_map = {
        "open_palm": "✋", "thumbs_up": "👍", "victory": "✌️",
        "fist": "✊", "pointing": "👉"
    }
    icon = icon_map.get(gesture, "🖐️") if gesture != "None" else "🤚"
    gesture_name_display = gesture.replace('_', ' ').title() if gesture != "None" else "No gesture"

    gesture_display.markdown(f"""
    <div class="gesture-card">
        <div style="font-size: 4rem;">{icon}</div>
        <div class="gesture-name">{gesture_name_display}</div>
    </div>
    """, unsafe_allow_html=True)

    progress_html = f"""
    <div class="progress-bar">
        <div class="progress-fill" style="width: {confidence}%;"></div>
    </div>
    <div class="confidence-text">Confidence: {confidence}%</div>
    """
    progress_placeholder.markdown(progress_html, unsafe_allow_html=True)

    total_detections.metric("Total Gestures", total_detected)
    last_gesture_time.metric("Last Gesture", gesture_name_display if gesture != "None" else "—")

    # Draw info on frame
    cv2.putText(frame, f"{gesture_name_display} ({confidence}%)", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return frame, last_saved, total_detected, current_gesture

# ----------------------------------------------------------------------
# LOCAL MODE (real‑time OpenCV)
# ----------------------------------------------------------------------
def run_local_mode():
    source = cam.get_camera_source()   # uses sidebar widgets
    cap = cam.open_camera(source, width, height)
    if cap is None:
        st.error("Could not open local camera. Please check your camera connection.")
        return

    with GestureRecognizer.create_from_options(options) as recognizer:
        # Clear queue
        while not detection_queue.empty():
            try:
                detection_queue.get_nowait()
            except:
                break

        last_saved = "None"
        frame_counter = 0
        total_detected = 0
        current_gesture = {"name": "None", "confidence": 0}

        try:
            while run_stream:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.02)
                    continue

                frame_counter += 1
                frame = cam.preprocess_frame(frame, source, width, height)

                frame, last_saved, total_detected, current_gesture = process_frame_and_update(
                    frame, recognizer, frame_counter, last_saved, total_detected, current_gesture
                )

                # Display frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

        except Exception as e:
            st.error(f"Stream error: {e}")
        finally:
            cam.release_camera(cap)
            frame_placeholder.empty()
            st.info("Camera stopped.")

# ----------------------------------------------------------------------
# CLOUD MODE (frame‑by‑frame using st.camera_input)
# ----------------------------------------------------------------------
def run_cloud_mode():
    st.info("🌐 Cloud mode: Click 'Capture' to take a photo. The app will detect gestures in that photo.")
    # Create a new recognizer (or reuse same options – but each frame we call synchronously)
    # We'll use the synchronous version for simplicity.
    
    # Use synchronous recognizer for cloud mode (easier)
    sync_options = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    
    with GestureRecognizer.create_from_options(sync_options) as recognizer:
        last_saved = "None"
        total_detected = 0
        current_gesture = {"name": "None", "confidence": 0}
        
        # Get a frame from the browser
        captured_img = st.camera_input("📸 Take a picture", key="cloud_camera")
        if captured_img is not None:
            # Convert to OpenCV BGR
            import numpy as np
            bytes_data = captured_img.getvalue()
            np_arr = np.frombuffer(bytes_data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            frame = cv2.resize(frame, (width, height))  # match resolution
            
            # Process single frame
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = recognizer.recognize(mp_image)
            
            # Parse result
            if result.gestures and len(result.gestures) > 0:
                top = result.gestures[0][0]
                gesture_name = top.category_name
                confidence = int(top.score * 100)
                if gesture_name != "None":
                    current_gesture = {"name": gesture_name, "confidence": confidence}
                    # Save to DB
                    if gesture_name != last_saved:
                        save_gesture(gesture_name, confidence)
                        last_saved = gesture_name
                        total_detected += 1
                        history_placeholder.dataframe(fetch_logs(), use_container_width=True, hide_index=True)
            else:
                current_gesture = {"name": "None", "confidence": 0}
            
            # Annotate frame
            gesture = current_gesture["name"]
            conf = current_gesture["confidence"]
            gesture_name_display = gesture.replace('_', ' ').title() if gesture != "None" else "No gesture"
            cv2.putText(frame, f"{gesture_name_display} ({conf}%)", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Update UI panels
            icon_map = {
                "open_palm": "✋", "thumbs_up": "👍", "victory": "✌️",
                "fist": "✊", "pointing": "👉"
            }
            icon = icon_map.get(gesture, "🖐️") if gesture != "None" else "🤚"
            gesture_display.markdown(f"""
            <div class="gesture-card">
                <div style="font-size: 4rem;">{icon}</div>
                <div class="gesture-name">{gesture_name_display}</div>
            </div>
            """, unsafe_allow_html=True)
            
            progress_html = f"""
            <div class="progress-bar">
                <div class="progress-fill" style="width: {conf}%;"></div>
            </div>
            <div class="confidence-text">Confidence: {conf}%</div>
            """
            progress_placeholder.markdown(progress_html, unsafe_allow_html=True)
            
            total_detections.metric("Total Gestures", total_detected)
            last_gesture_time.metric("Last Gesture", gesture_name_display if gesture != "None" else "—")
            
            # Display the captured frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            
            # Refresh button to allow another capture
            if st.button("📸 Capture another"):
                st.rerun()
        else:
            st.info("Click the camera button above to take a picture.")

# ============================================
# MAIN: choose mode based on environment
# ============================================
if run_stream:
    history_placeholder.dataframe(fetch_logs(), use_container_width=True, hide_index=True)
    if cam.is_cloud():
        run_cloud_mode()
    else:
        run_local_mode()
else:
    st.info("Click **START CAMERA** to begin gesture recognition.")
    history_placeholder.dataframe(fetch_logs(), use_container_width=True, hide_index=True)