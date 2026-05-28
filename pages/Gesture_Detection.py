# import os
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# import cv2
# import mediapipe as mp
# import time
# import sqlite3
# import streamlit as st
# import pandas as pd
# from datetime import datetime
# from utils import apply_global_style
# from camera_utils import get_camera_source, open_camera, preprocess_frame

# st.set_page_config(page_title="Gesture Detection", page_icon="✋", layout="wide")
# apply_global_style()

# DB_FILE = "gesture_history.db"

# def init_db():
#     conn = sqlite3.connect(DB_FILE)
#     conn.execute('''CREATE TABLE IF NOT EXISTS gesture_logs
#         (id INTEGER PRIMARY KEY, gesture_name TEXT, confidence_percentage TEXT, time_of_detection TEXT, date_of_detection TEXT)''')
#     conn.close()

# def save_gesture(name, conf):
#     now = datetime.now()
#     conn = sqlite3.connect(DB_FILE)
#     conn.execute("INSERT INTO gesture_logs (gesture_name, confidence_percentage, time_of_detection, date_of_detection) VALUES (?,?,?,?)",
#                  (name, f"{conf}%", now.strftime("%H:%M:%S"), now.strftime("%Y-%m-%d")))
#     conn.commit()
#     conn.close()

# def fetch_logs():
#     conn = sqlite3.connect(DB_FILE)
#     df = pd.read_sql_query("SELECT * FROM gesture_logs ORDER BY id DESC LIMIT 50", conn)
#     conn.close()
#     return df

# init_db()

# # Load MediaPipe model
# model_path = 'D:/Data_science/visionmateai_v1 - Copy - Copy/pages/gesture_recognizer.task'
# if not os.path.exists(model_path):
#     st.error("Model file 'gesture_recognizer.task' not found.")
#     st.stop()

# BaseOptions = mp.tasks.BaseOptions
# GestureRecognizer = mp.tasks.vision.GestureRecognizer
# GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
# VisionRunningMode = mp.tasks.vision.RunningMode

# latest_gesture = {"name": "None", "confidence": 0}
# def callback(result, output_image, timestamp_ms):
#     global latest_gesture
#     if result.gestures and len(result.gestures) > 0:
#         top = result.gestures[0][0]
#         latest_gesture = {"name": top.category_name, "confidence": int(top.score * 100)}
#     else:
#         latest_gesture = {"name": "None", "confidence": 0}

# options = GestureRecognizerOptions(
#     base_options=BaseOptions(model_asset_path=model_path),
#     running_mode=VisionRunningMode.LIVE_STREAM,
#     num_hands=1,
#     result_callback=callback
# )

# st.title("✋ Real-Time Gesture Recognition")
# st.markdown("Recognizes: thumbs_up, victory, open_palm, fist, pointing")

# # Sidebar
# camera_source = get_camera_source()
# resolution = st.sidebar.selectbox("Quality Profile", ["Low (Fastest)", "Medium (Balanced)"])
# width, height = (320, 240) if resolution == "Low (Fastest)" else (640, 480)

# run_stream = st.checkbox("🔄 Start Video Stream", value=False)
# status_placeholder = st.empty()
# frame_placeholder = st.empty()
# history_placeholder = st.empty()

# def start_streaming(source, w, h):
#     cap = open_camera(source, w, h)
#     if not cap or not cap.isOpened():
#         st.error("Cannot open camera")
#         return
    
#     with GestureRecognizer.create_from_options(options) as recognizer:
#         last_saved = "None"
#         while run_stream:
#             ret, frame = cap.read()
#             if not ret:
#                 time.sleep(0.05)
#                 continue
            
#             frame = preprocess_frame(frame, source, w, h)
#             frame = cv2.flip(frame, 1)
#             rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
#             recognizer.recognize_async(mp_image, int(time.time() * 1000))
            
#             gesture = latest_gesture["name"]
#             confidence = latest_gesture["confidence"]
            
#             # Save unique gestures
#             if gesture != "None" and gesture != last_saved:
#                 save_gesture(gesture, confidence)
#                 last_saved = gesture
#                 history_placeholder.dataframe(fetch_logs(), use_container_width=True, hide_index=True)
            
#             # Display on frame
#             cv2.putText(frame, f"{gesture} ({confidence}%)", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
#             status_placeholder.markdown(f"**Gesture:** `{gesture}` | **Confidence:** `{confidence}%`")
            
#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=False, width=w)
#             time.sleep(0.01)
    
#     cap.release()

# if run_stream:
#     history_placeholder.dataframe(fetch_logs(), use_container_width=True, hide_index=True)
#     start_streaming(camera_source, width, height)
# else:
#     st.write("Stopped. Check the box above to start streaming.")
#     history_placeholder.dataframe(fetch_logs(), use_container_width=True, hide_index=True)
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
from utils import apply_global_style
from camera_utils import get_camera_source, open_camera, preprocess_frame, release_camera

# ============================================
# PAGE CONFIG & ATTRACTIVE DARK THEME (Soft, readable)
# ============================================
st.set_page_config(page_title="Gesture Detection", page_icon="✋", layout="wide")
apply_global_style()

# ============================================
# DATABASE SETUP (unchanged)
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
# MEDIAPIPE SETUP (unchanged)
# ============================================
model_path = 'pages/models/gesture_recognizer.task'
#model_path = 'E:\\FinalvisionMateAi-V2\\FinalvisionMateAi-V2\\pages\\models\\gesture_recognizer.task'   # Update to your actual path
if not os.path.exists(model_path):
    st.error("Model file 'gesture_recognizer.task' not found.")
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
# CAMERA & SETTINGS (unchanged)
# ============================================
camera_source = get_camera_source()

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

# Layout: two columns
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
# STREAMING FUNCTION (no changes)
# ============================================
def start_streaming(source, w, h, process_interval):
    cap = open_camera(source, w, h)
    if cap is None:
        st.error(f"Cannot open camera source: {source}. Try a different option.")
        return

    with GestureRecognizer.create_from_options(options) as recognizer:
        while not detection_queue.empty():
            try:
                detection_queue.get_nowait()
            except:
                break

        last_saved = "None"
        frame_counter = 0
        current_gesture = {"name": "None", "confidence": 0}
        total_detected = 0

        try:
            while run_stream:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.02)
                    continue

                frame_counter += 1
                frame = preprocess_frame(frame, source, w, h)

                if frame_counter % process_interval == 0:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    recognizer.recognize_async(mp_image, frame_counter)

                try:
                    gesture_name, confidence = detection_queue.get_nowait()
                    current_gesture = {"name": gesture_name, "confidence": confidence}
                except:
                    pass

                gesture = current_gesture["name"]
                confidence = current_gesture["confidence"]

                if gesture != "None" and gesture != last_saved:
                    save_gesture(gesture, confidence)
                    last_saved = gesture
                    total_detected += 1
                    history_placeholder.dataframe(fetch_logs(), use_container_width=True, hide_index=True)

                # --- UI updates (theme-aware) ---
                icon = "✋" if gesture == "open_palm" else "👍" if gesture == "thumbs_up" else "✌️" if gesture == "victory" else "✊" if gesture == "fist" else "👉" if gesture == "pointing" else "🖐️"
                if gesture != "None":
                    gesture_display.markdown(f"""
                    <div class="gesture-card">
                        <div style="font-size: 4rem;">{icon}</div>
                        <div class="gesture-name">{gesture.replace('_', ' ').title()}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    gesture_display.markdown(f"""
                    <div class="gesture-card">
                        <div style="font-size: 4rem;">🤚</div>
                        <div class="gesture-name">No gesture</div>
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
                if gesture != "None":
                    last_gesture_time.metric("Last Gesture", gesture.replace('_', ' ').title())
                else:
                    last_gesture_time.metric("Last Gesture", "—")

                # Draw info on frame
                cv2.putText(frame, f"{gesture} ({confidence}%)", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

        except Exception as e:
            st.error(f"Stream error: {e}")
        finally:
            release_camera(cap)
            frame_placeholder.empty()
            gesture_display.empty()
            progress_placeholder.empty()
            total_detections.empty()
            last_gesture_time.empty()
            st.info("Camera stopped.")

if run_stream:
    history_placeholder.dataframe(fetch_logs(), use_container_width=True, hide_index=True)
    start_streaming(camera_source, width, height, process_every_n)
else:
    st.info("Click **START CAMERA** to begin real-time gesture recognition.")
    history_placeholder.dataframe(fetch_logs(), use_container_width=True, hide_index=True)