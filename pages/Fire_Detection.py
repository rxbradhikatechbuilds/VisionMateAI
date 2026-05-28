# import streamlit as st
# import cv2
# import numpy as np
# import requests
# import pandas as pd
# import plotly.graph_objects as go
# import plotly.express as px
# import time
# from datetime import datetime
# import base64
# from PIL import Image
# import warnings
# warnings.filterwarnings('ignore')

# # ============================================
# # PAGE CONFIG & GLOBAL STYLE (Dark Theme)
# # ============================================
# st.set_page_config(
#     page_title="Fire Detection System",
#     page_icon="🔥",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# def apply_global_style():
#     st.markdown("""
#     <style>
#         .stApp { background-color: #0a0a0a; }
#         .css-1d391kg, .css-12oz5g7 { background-color: #1e1e1e; }
#         [data-testid="stMetricValue"] { color: #ff4444 !important; font-size: 2rem !important; }
#         [data-testid="stMetricLabel"] { color: #cccccc !important; }
#         h1, h2, h3, .stMarkdown { color: #ffffff !important; }
#         .stButton > button {
#             background-color: #333333; color: white; border-radius: 8px;
#             border: 1px solid #ff4444; transition: 0.3s;
#         }
#         .stButton > button:hover { background-color: #ff4444; color: black; border-color: white; }
#         .stAlert { border-radius: 10px; }
#         .dataframe { background-color: #1e1e1e !important; color: white !important; }
#     </style>
#     """, unsafe_allow_html=True)

# apply_global_style()

# # ============================================
# # CONFIGURATION
# # ============================================
# API_URL = "http://localhost:8000"
# PREDICT_ENDPOINT = f"{API_URL}/predict"
# HEALTH_ENDPOINT = f"{API_URL}/health"
# THRESHOLDS_ENDPOINT = f"{API_URL}/thresholds"

# # Session state initialization
# if 'detection_history' not in st.session_state:
#     st.session_state.detection_history = []
# if 'stats' not in st.session_state:
#     st.session_state.stats = {
#         'total_detections': 0, 'fire_detected': 0, 'no_fire': 0,
#         'uncertain': 0, 'undetermined': 0, 'avg_confidence': 0, 'total_processing_time': 0
#     }

# # ============================================
# # HELPER FUNCTIONS
# # ============================================
# def predict_image(image_bytes):
#     """Send image to FastAPI and return result"""
#     try:
#         files = {'file': ('image.jpg', image_bytes, 'image/jpeg')}
#         response = requests.post(PREDICT_ENDPOINT, files=files, timeout=30)
#         if response.status_code == 200:
#             return response.json()
#     except Exception:
#         pass
#     return None

# def draw_detections(frame, detections):
#     """Draw bounding boxes (red=high, orange=medium, yellow=low)"""
#     if not detections:
#         return frame
#     for det in detections:
#         bbox = det['bbox']
#         conf = det['confidence']
#         class_name = det['class_name']
#         x1, y1, x2, y2 = map(int, bbox)
#         color = (0, 0, 255) if conf > 0.7 else (0, 165, 255) if conf > 0.4 else (0, 255, 255)
#         cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#         label = f"{class_name}: {conf*100:.1f}%"
#         cv2.rectangle(frame, (x1, y1-25), (x1+len(label)*10, y1), color, -1)
#         cv2.putText(frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
#     return frame

# def update_statistics(result):
#     """Update session stats and history"""
#     status = result.get('status')
#     confidence = result.get('max_confidence', 0)
#     processing_time = result.get('processing_time_ms', 0)
    
#     st.session_state.stats['total_detections'] += 1
#     st.session_state.stats[status] = st.session_state.stats.get(status, 0) + 1
#     st.session_state.stats['total_processing_time'] += processing_time
    
#     total_conf = st.session_state.stats['avg_confidence'] * (st.session_state.stats['total_detections'] - 1)
#     st.session_state.stats['avg_confidence'] = (total_conf + confidence) / st.session_state.stats['total_detections']
    
#     st.session_state.detection_history.append({
#         'timestamp': datetime.now(), 'status': status, 'confidence': confidence,
#         'processing_time': processing_time, 'num_detections': result.get('num_detections', 0)
#     })
#     if len(st.session_state.detection_history) > 500:
#         st.session_state.detection_history = st.session_state.detection_history[-500:]

# def check_api_health():
#     try:
#         r = requests.get(HEALTH_ENDPOINT, timeout=3)
#         return r.status_code == 200 and r.json().get('model_loaded', False)
#     except:
#         return False

# # ============================================
# # PAGE: LIVE CAMERA (FIXED - NO THREADING)
# # ============================================
# def live_camera_page():
#     st.markdown("## 🎥 Live Camera Fire Detection")
    
#     if not check_api_health():
#         st.error("❌ API is not reachable. Please start the backend: `python app.py`")
#         return
    
#     # Camera settings
#     col1, col2, col3 = st.columns(3)
#     with col1:
#         camera_source = st.selectbox("Camera Source", ["0 (Built-in)", "1 (External)", "2 (USB)"])
#         camera_id = int(camera_source.split()[0])
#     with col2:
#         resolution = st.selectbox("Resolution", ["Low (320x240)", "Medium (640x480)"])
#         w, h = (320, 240) if "Low" in resolution else (640, 480)
#     with col3:
#         process_every_n = st.slider("Process every N frames", 1, 10, 3,
#                                     help="Higher = faster but less accurate")
    
#     run_stream = st.checkbox("🔄 START CAMERA", value=False)
    
#     # Placeholders for dynamic content
#     frame_placeholder = st.empty()
#     metrics_placeholder = st.empty()
#     status_placeholder = st.empty()
    
#     if run_stream:
#         # Open camera
#         cap = cv2.VideoCapture(camera_id)
#         if not cap.isOpened():
#             st.error(f"Could not open camera {camera_id}. Try a different source.")
#             return
        
#         # Set resolution
#         cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
#         cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        
#         frame_count = 0
#         st.success(f"Camera {camera_id} opened. Streaming...")
        
#         try:
#             while run_stream:
#                 ret, frame = cap.read()
#                 if not ret:
#                     time.sleep(0.02)
#                     continue
                
#                 frame_count += 1
                
#                 # Process every N frames
#                 if frame_count % process_every_n == 0:
#                     # Encode and send to API
#                     _, buffer = cv2.imencode('.jpg', frame)
#                     result = predict_image(buffer.tobytes())
                    
#                     if result:
#                         update_statistics(result)
#                         # Draw detections
#                         if result.get('detections'):
#                             frame = draw_detections(frame, result['detections'])
                        
#                         # Update metrics display
#                         with metrics_placeholder.container():
#                             col1, col2, col3, col4 = st.columns(4)
#                             with col1:
#                                 st.metric("Status", result['status'].replace('_', ' ').title())
#                             with col2:
#                                 conf = result['max_confidence'] * 100
#                                 st.metric("Confidence", f"{conf:.1f}%")
#                             with col3:
#                                 st.metric("Detections", result['num_detections'])
#                             with col4:
#                                 st.metric("API Time", f"{result['processing_time_ms']:.0f}ms")
                        
#                         # Alert messages
#                         if result['status'] == 'fire_detected':
#                             status_placeholder.error(f"🚨 FIRE DETECTED! Confidence: {conf:.1f}%")
#                         elif result['status'] == 'uncertain':
#                             status_placeholder.warning(f"⚠️ Possible Fire: {conf:.1f}% confidence")
#                         else:
#                             status_placeholder.success("✅ No Fire Detected")
#                     else:
#                         status_placeholder.error("API not responding")
                
#                 # Show the frame (with or without boxes)
#                 frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                
#                 time.sleep(0.033)  # ~30 fps
#         except Exception as e:
#             st.error(f"Stream error: {e}")
#         finally:
#             cap.release()
#             frame_placeholder.empty()
#             metrics_placeholder.empty()
#             status_placeholder.empty()
#             st.info("Camera stopped.")
#     else:
#         st.info("Press START CAMERA to begin real-time detection.")

# # ============================================
# # PAGE: IMAGE UPLOAD
# # ============================================
# def image_upload_page():
#     st.markdown("## 📤 Image Upload Detection")
    
#     if not check_api_health():
#         st.error("❌ API not reachable. Start backend first.")
#         return
    
#     uploaded_files = st.file_uploader("Choose images...", type=['jpg','jpeg','png','bmp'], accept_multiple_files=True)
    
#     if uploaded_files:
#         col1, col2 = st.columns([2,1])
#         with col1:
#             selected = st.selectbox("Select image", [f.name for f in uploaded_files])
#             file = [f for f in uploaded_files if f.name == selected][0]
#             image = Image.open(file)
#             st.image(image, caption=selected, use_container_width=True)
#         with col2:
#             if st.button("🔍 Detect Fire", type="primary", use_container_width=True):
#                 with st.spinner("Processing..."):
#                     result = predict_image(file.getvalue())
#                     if result:
#                         update_statistics(result)
#                         st.markdown("### Results")
#                         if result['status'] == 'fire_detected':
#                             st.error(f"🔥 FIRE DETECTED!")
#                         elif result['status'] == 'uncertain':
#                             st.warning(f"⚠️ Possible Fire")
#                         else:
#                             st.success(f"✅ No Fire")
#                         c1,c2,c3 = st.columns(3)
#                         c1.metric("Confidence", f"{result['max_confidence']*100:.1f}%")
#                         c2.metric("Detections", result['num_detections'])
#                         c3.metric("Time", f"{result['processing_time_ms']:.0f}ms")
#                         if result.get('detections'):
#                             st.write("**Detections:**")
#                             for d in result['detections']:
#                                 st.write(f"- {d['class_name']}: {d['confidence']*100:.1f}%")
        
#         if st.button("Process All Images", use_container_width=True):
#             with st.spinner(f"Processing {len(uploaded_files)} images..."):
#                 for f in uploaded_files:
#                     r = predict_image(f.getvalue())
#                     if r:
#                         update_statistics(r)
#                 st.success("All images processed!")
#                 st.rerun()

# # ============================================
# # PAGE: DASHBOARD & EXPORTS
# # ============================================
# def dashboard_page():
#     st.markdown("## 📊 Detection Analytics Dashboard")
    
#     if not st.session_state.detection_history:
#         st.info("No data yet. Start camera or upload images.")
#         return
    
#     df = pd.DataFrame(st.session_state.detection_history)
    
#     # Metrics
#     col1,col2,col3,col4,col5 = st.columns(5)
#     with col1:
#         st.metric("Total Detections", len(df))
#     with col2:
#         fire = len(df[df['status']=='fire_detected'])
#         st.metric("🔥 Fire Events", fire)
#     with col3:
#         st.metric("Avg Confidence", f"{df['confidence'].mean()*100:.1f}%")
#     with col4:
#         st.metric("Avg Processing", f"{df['processing_time'].mean():.0f}ms")
#     with col5:
#         rate = (fire/len(df))*100 if len(df)>0 else 0
#         st.metric("Fire Rate", f"{rate:.1f}%")
    
#     # Charts
#     c1,c2 = st.columns(2)
#     with c1:
#         fig_pie = px.pie(values=df['status'].value_counts().values,
#                          names=df['status'].value_counts().index,
#                          title="Status Distribution",
#                          color_discrete_sequence=['#ff4444','#ffaa44','#44ff44','#888888'])
#         fig_pie.update_layout(paper_bgcolor='#1e1e1e', font_color='white')
#         st.plotly_chart(fig_pie, use_container_width=True)
#     with c2:
#         fig_line = go.Figure()
#         fig_line.add_trace(go.Scatter(x=df['timestamp'], y=df['confidence']*100,
#                                       mode='lines+markers', name='Confidence',
#                                       line=dict(color='#ff4444', width=2)))
#         fig_line.add_hline(y=35, line_dash="dash", line_color='orange', annotation_text="Threshold 35%")
#         fig_line.update_layout(title="Confidence Trend", xaxis_title="Time", yaxis_title="Confidence (%)",
#                                paper_bgcolor='#1e1e1e', font_color='white', plot_bgcolor='#0a0a0a')
#         fig_line.update_xaxes(gridcolor='#333')
#         fig_line.update_yaxes(gridcolor='#333')
#         st.plotly_chart(fig_line, use_container_width=True)
    
#     # Recent table
#     st.markdown("### Recent Detections")
#     recent = df.tail(10).sort_values('timestamp', ascending=False)
#     recent['timestamp'] = recent['timestamp'].dt.strftime('%H:%M:%S')
#     recent['confidence'] = recent['confidence'].apply(lambda x: f"{x*100:.1f}%")
#     st.dataframe(recent[['timestamp','status','confidence','num_detections','processing_time']],
#                  use_container_width=True)
    
#     # Export / Clear
#     col1,col2 = st.columns(2)
#     with col1:
#         if st.button("📥 Export to CSV", use_container_width=True):
#             csv = df.to_csv(index=False)
#             b64 = base64.b64encode(csv.encode()).decode()
#             href = f'<a href="data:file/csv;base64,{b64}" download="fire_report.csv">Download CSV</a>'
#             st.markdown(href, unsafe_allow_html=True)
#     with col2:
#         if st.button("🗑️ Clear History", use_container_width=True):
#             st.session_state.detection_history = []
#             st.session_state.stats = {'total_detections':0,'fire_detected':0,'no_fire':0,
#                                       'uncertain':0,'undetermined':0,'avg_confidence':0,'total_processing_time':0}
#             st.success("Cleared!")
#             st.rerun()

# # ============================================
# # PAGE: SETTINGS
# # ============================================
# def settings_page():
#     st.markdown("## ⚙️ Settings")
#     st.markdown(f"**API URL:** `{API_URL}`")
#     try:
#         r = requests.get(THRESHOLDS_ENDPOINT, timeout=2)
#         if r.status_code == 200:
#             t = r.json()
#             col1,col2 = st.columns(2)
#             col1.metric("Confidence Threshold", f"{t.get('confidence_threshold',0.5)*100:.0f}%")
#             col1.metric("Ambiguous Threshold", f"{t.get('ambiguous_threshold',0.35)*100:.0f}%")
#             col2.metric("IoU Threshold", f"{t.get('iou_threshold',0.45)*100:.0f}%")
#             col2.metric("Low Confidence", f"{t.get('low_confidence_threshold',0.25)*100:.0f}%")
#         else:
#             st.warning("Could not fetch thresholds from API")
#     except:
#         st.error("API not reachable")
    
#     st.markdown("### System Info")
#     st.info(f"Streamlit version: {st.__version__}")
#     st.info(f"History records: {len(st.session_state.detection_history)}")
#     st.info(f"Total detections: {st.session_state.stats['total_detections']}")

# # ============================================
# # MAIN APP
# # ============================================
# def main():
#     st.title("🔥 Real-Time Fire Detection System")
#     st.markdown("Powered by **YOLOv8 + FastAPI** | Dark Theme")
    
#     # Sidebar
#     st.sidebar.markdown("## Navigation")
#     page = st.sidebar.radio("Go to", ["🎥 Live Camera", "📤 Upload Images", "📊 Dashboard", "⚙️ Settings"])
    
#     st.sidebar.markdown("---")
#     st.sidebar.markdown("### API Status")
#     if check_api_health():
#         st.sidebar.success("✅ Connected")
#         try:
#             info = requests.get(HEALTH_ENDPOINT).json()
#             st.sidebar.info(f"Device: {info.get('device','unknown')}")
#         except:
#             pass
#     else:
#         st.sidebar.error("❌ Not reachable")
#         st.sidebar.warning("Run: `python app.py`")
    
#     st.sidebar.markdown("---")
#     st.sidebar.markdown("### Session Stats")
#     st.sidebar.metric("Total Detections", st.session_state.stats['total_detections'])
#     st.sidebar.metric("🔥 Fire Events", st.session_state.stats['fire_detected'])
#     st.sidebar.metric("Avg Confidence", f"{st.session_state.stats['avg_confidence']*100:.1f}%")
    
#     st.sidebar.markdown("---")
#     st.sidebar.info("1. Start backend\n2. Select camera\n3. Press START CAMERA")
    
#     # Page router
#     if page == "🎥 Live Camera":
#         live_camera_page()
#     elif page == "📤 Upload Images":
#         image_upload_page()
#     elif page == "📊 Dashboard":
#         dashboard_page()
#     elif page == "⚙️ Settings":
#         settings_page()

# if __name__ == "__main__":
#     main()

import streamlit as st
import cv2
import numpy as np
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import time
from datetime import datetime
import base64
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

from utils import apply_global_style
from camera_utils import get_camera_source, open_camera, preprocess_frame, release_camera

# Apply adaptive theme logic
apply_global_style()

# ============================================
# PAGE CONFIG & ATTRACTIVE DARK THEME (Soft, readable)
# ============================================
st.set_page_config(
    page_title="Fire Detection System",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CONFIGURATION (unchanged)
# ============================================
# API_URL = "http://localhost:8000"
API_URL = "https://visionmateai-1.onrender.com"
PREDICT_ENDPOINT = f"{API_URL}/predict"
HEALTH_ENDPOINT = f"{API_URL}/health"
THRESHOLDS_ENDPOINT = f"{API_URL}/thresholds"

if 'detection_history' not in st.session_state:
    st.session_state.detection_history = []
if 'stats' not in st.session_state:
    st.session_state.stats = {
        'total_detections': 0, 'fire_detected': 0, 'no_fire': 0,
        'uncertain': 0, 'undetermined': 0, 'avg_confidence': 0, 'total_processing_time': 0
    }

# ============================================
# HELPER FUNCTIONS (unchanged)
# ============================================
def predict_image(image_bytes):
    try:
        print("PREDICT_ENDPOINT:-", PREDICT_ENDPOINT)
        files = {'file': ('image.jpg', image_bytes, 'image/jpeg')}
        response = requests.post(PREDICT_ENDPOINT, files=files, timeout=30)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None

def draw_detections(frame, detections):
    if not detections:
        return frame
    for det in detections:
        bbox = det['bbox']
        conf = det['confidence']
        class_name = det['class_name']
        x1, y1, x2, y2 = map(int, bbox)
        color = (0, 0, 255) if conf > 0.7 else (0, 165, 255) if conf > 0.4 else (0, 255, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{class_name}: {conf*100:.1f}%"
        if y1 > 25:
            cv2.putText(frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        else:
            cv2.putText(frame, label, (x1, y1+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    return frame

def update_statistics(result):
    status = result.get('status')
    confidence = result.get('max_confidence', 0)
    processing_time = result.get('processing_time_ms', 0)
    st.session_state.stats['total_detections'] += 1
    st.session_state.stats[status] = st.session_state.stats.get(status, 0) + 1
    st.session_state.stats['total_processing_time'] += processing_time
    total_conf = st.session_state.stats['avg_confidence'] * (st.session_state.stats['total_detections'] - 1)
    st.session_state.stats['avg_confidence'] = (total_conf + confidence) / st.session_state.stats['total_detections']
    st.session_state.detection_history.append({
        'timestamp': datetime.now(), 'status': status, 'confidence': confidence,
        'processing_time': processing_time, 'num_detections': result.get('num_detections', 0)
    })
    if len(st.session_state.detection_history) > 500:
        st.session_state.detection_history = st.session_state.detection_history[-500:]

def check_api_health():
    try:
        print("Health API Endpoint:- ", HEALTH_ENDPOINT)
        r = requests.get(HEALTH_ENDPOINT, timeout=3)
        return r.status_code == 200 and r.json().get('model_loaded', False)
    except:
        return False

# ============================================
# LIVE CAMERA PAGE (theme only)
# ============================================
def live_camera_page():
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("🎥 Live Camera Fire Detection (Fast Mode)")
    
    if not check_api_health():
        st.error("❌ API is not reachable. Please start the backend: `python app.py`")
        return
    
    camera_source = get_camera_source()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        resolution = st.selectbox("Resolution (lower = faster)", 
                                  ["Very Low (160x120)", "Low (320x240)", "Medium (480x360)"])
        if resolution == "Very Low (160x120)":
            w, h = 160, 120
        elif resolution == "Low (320x240)":
            w, h = 320, 240
        else:
            w, h = 480, 360
    with col2:
        process_every_n = st.slider("Process every N frames (higher = faster)", 1, 15, 5)
    with col3:
        draw_boxes = st.checkbox("Draw detection boxes", value=True)
        jpeg_quality = st.slider("JPEG Quality (lower = faster upload)", 30, 95, 70, step=5)
    
    run_stream = st.checkbox("🔄 START CAMERA (Fast Mode)", value=False)
    
    frame_placeholder = st.empty()
    metrics_placeholder = st.empty()
    status_placeholder = st.empty()
    
    if run_stream:
        cap = open_camera(camera_source, w, h)
        if cap is None:
            st.error(f"Could not open camera source: {camera_source}")
            return
        
        st.success(f"Camera opened at {w}x{h}. Streaming...")
        frame_count = 0
        try:
            while run_stream:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.02)
                    continue
                frame = preprocess_frame(frame, camera_source, w, h)
                frame_count += 1
                
                if frame_count % process_every_n == 0:
                    encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
                    _, buffer = cv2.imencode('.jpg', frame, encode_params)
                    result = predict_image(buffer.tobytes())
                    
                    if result:
                        update_statistics(result)
                        if draw_boxes and result.get('detections'):
                            frame = draw_detections(frame, result['detections'])
                        
                        with metrics_placeholder.container():
                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("Status", result['status'].replace('_', ' ').title())
                            conf = result['max_confidence'] * 100
                            col2.metric("Confidence", f"{conf:.1f}%")
                            col3.metric("Detections", result['num_detections'])
                            col4.metric("API Time", f"{result['processing_time_ms']:.0f}ms")
                        
                        if result['status'] == 'fire_detected':
                            status_placeholder.error(f"🚨 FIRE DETECTED! Confidence: {conf:.1f}%")
                        elif result['status'] == 'uncertain':
                            status_placeholder.warning(f"⚠️ Possible Fire: {conf:.1f}%")
                        else:
                            status_placeholder.success("✅ No Fire Detected")
                    else:
                        status_placeholder.error("API not responding")
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
        except Exception as e:
            st.error(f"Stream error: {e}")
        finally:
            release_camera(cap)
            frame_placeholder.empty()
            metrics_placeholder.empty()
            status_placeholder.empty()
            st.info("Camera stopped.")
    else:
        st.info("Press START CAMERA to begin real-time detection (fast mode).")
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================
# IMAGE UPLOAD PAGE (unchanged)
# ============================================
def image_upload_page():
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("📤 Image Upload Detection")
    if not check_api_health():
        st.error("❌ API not reachable. Start backend first.")
        return
    uploaded_files = st.file_uploader("Choose images...", type=['jpg','jpeg','png','bmp'], accept_multiple_files=True)
    if uploaded_files:
        col1, col2 = st.columns([2,1])
        with col1:
            selected = st.selectbox("Select image", [f.name for f in uploaded_files])
            file = [f for f in uploaded_files if f.name == selected][0]
            image = Image.open(file)
            st.image(image, caption=selected, use_container_width=True)
        with col2:
            if st.button("🔍 Detect Fire", type="primary", use_container_width=True):
                with st.spinner("Processing..."):
                    result = predict_image(file.getvalue())
                    if result:
                        update_statistics(result)
                        st.markdown("### Results")
                        if result['status'] == 'fire_detected':
                            st.error("🔥 FIRE DETECTED!")
                        elif result['status'] == 'uncertain':
                            st.warning("⚠️ Possible Fire")
                        else:
                            st.success("✅ No Fire")
                        c1,c2,c3 = st.columns(3)
                        c1.metric("Confidence", f"{result['max_confidence']*100:.1f}%")
                        c2.metric("Detections", result['num_detections'])
                        c3.metric("Time", f"{result['processing_time_ms']:.0f}ms")
                        if result.get('detections'):
                            st.write("**Detections:**")
                            for d in result['detections']:
                                st.write(f"- {d['class_name']}: {d['confidence']*100:.1f}%")
        if st.button("Process All Images", use_container_width=True):
            with st.spinner(f"Processing {len(uploaded_files)} images..."):
                for f in uploaded_files:
                    r = predict_image(f.getvalue())
                    if r:
                        update_statistics(r)
                st.success("All images processed!")
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================
# DASHBOARD PAGE (with dark-themed charts)
# ============================================
def dashboard_page():
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("📊 Detection Analytics Dashboard")
    if not st.session_state.detection_history:
        st.info("No data yet. Start camera or upload images.")
        return
    df = pd.DataFrame(st.session_state.detection_history)
    
    col1,col2,col3,col4,col5 = st.columns(5)
    with col1:
        st.metric("Total Detections", len(df))
    with col2:
        fire = len(df[df['status']=='fire_detected'])
        st.metric("🔥 Fire Events", fire)
    with col3:
        st.metric("Avg Confidence", f"{df['confidence'].mean()*100:.1f}%")
    with col4:
        st.metric("Avg Processing", f"{df['processing_time'].mean():.0f}ms")
    with col5:
        rate = (fire/len(df))*100 if len(df)>0 else 0
        st.metric("Fire Rate", f"{rate:.1f}%")
    
    c1,c2 = st.columns(2)
    with c1:
        fig_pie = px.pie(values=df['status'].value_counts().values,
                         names=df['status'].value_counts().index,
                         title="Status Distribution",
                         color_discrete_sequence=['#ff4444','#ffaa44','#44ff44','#888888'])
        fig_pie.update_layout(paper_bgcolor='#23263b', font_color='#eef2ff')
        st.plotly_chart(fig_pie, use_container_width=True)
    with c2:
        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(x=df['timestamp'], y=df['confidence']*100,
                                      mode='lines+markers', name='Confidence',
                                      line=dict(color='#ff4444', width=2)))
        fig_line.add_hline(y=35, line_dash="dash", line_color='orange', annotation_text="Threshold 35%")
        fig_line.update_layout(title="Confidence Trend", xaxis_title="Time", yaxis_title="Confidence (%)",
                               paper_bgcolor='#23263b', font_color='#eef2ff', plot_bgcolor='#1a1c2e')
        fig_line.update_xaxes(gridcolor='#2c2f42')
        fig_line.update_yaxes(gridcolor='#2c2f42')
        st.plotly_chart(fig_line, use_container_width=True)
    
    st.markdown("### Recent Detections")
    recent = df.tail(10).sort_values('timestamp', ascending=False)
    recent['timestamp'] = recent['timestamp'].dt.strftime('%H:%M:%S')
    recent['confidence'] = recent['confidence'].apply(lambda x: f"{x*100:.1f}%")
    st.dataframe(recent[['timestamp','status','confidence','num_detections','processing_time']],
                 use_container_width=True)
    
    col1,col2 = st.columns(2)
    with col1:
        if st.button("📥 Export to CSV", use_container_width=True):
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="fire_report.csv">Download CSV</a>'
            st.markdown(href, unsafe_allow_html=True)
    with col2:
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.detection_history = []
            st.session_state.stats = {'total_detections':0,'fire_detected':0,'no_fire':0,
                                      'uncertain':0,'undetermined':0,'avg_confidence':0,'total_processing_time':0}
            st.success("Cleared!")
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================
# SETTINGS PAGE (unchanged)
# ============================================
def settings_page():
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("⚙️ Settings")
    st.markdown(f"**API URL:** `{API_URL}`")
    try:
        print("THRESHOLDS_ENDPOINT:- ", )
        r = requests.get(THRESHOLDS_ENDPOINT, timeout=2)
        if r.status_code == 200:
            t = r.json()
            col1,col2 = st.columns(2)
            col1.metric("Confidence Threshold", f"{t.get('confidence_threshold',0.5)*100:.0f}%")
            col1.metric("Ambiguous Threshold", f"{t.get('ambiguous_threshold',0.35)*100:.0f}%")
            col2.metric("IoU Threshold", f"{t.get('iou_threshold',0.45)*100:.0f}%")
            col2.metric("Low Confidence", f"{t.get('low_confidence_threshold',0.25)*100:.0f}%")
        else:
            st.warning("Could not fetch thresholds from API")
    except:
        st.error("API not reachable")
    st.markdown("### System Info")
    st.info(f"Streamlit version: {st.__version__}")
    st.info(f"History records: {len(st.session_state.detection_history)}")
    st.info(f"Total detections: {st.session_state.stats['total_detections']}")
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================
# MAIN
# ============================================
def main():
    st.title("🔥 Real-Time Fire Detection System")
    st.markdown("Powered by **YOLOv8 + FastAPI**")
    
    st.sidebar.markdown("## Navigation")
    page = st.sidebar.radio("Go to", ["🎥 Live Camera", "📤 Upload Images", "📊 Dashboard", "⚙️ Settings"])
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### API Status")
    if check_api_health():
        st.sidebar.success("✅ Connected")
        try:
            print("HEALTH_ENDPOINT:- ", HEALTH_ENDPOINT)
            info = requests.get(HEALTH_ENDPOINT).json()
            st.sidebar.info(f"Device: {info.get('device','unknown')}")
        except:
            pass
    else:
        st.sidebar.error("❌ Not reachable")
        st.sidebar.warning("Run: `python app.py`")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Session Stats")
    st.sidebar.metric("Total Detections", st.session_state.stats['total_detections'])
    st.sidebar.metric("🔥 Fire Events", st.session_state.stats['fire_detected'])
    st.sidebar.metric("Avg Confidence", f"{st.session_state.stats['avg_confidence']*100:.1f}%")
    
    st.sidebar.markdown("---")
    st.sidebar.info("⚡ Speed tips:\n- Lower resolution\n- Higher 'Process every N frames'\n- Lower JPEG quality\n- Disable drawing boxes")
    
    if page == "🎥 Live Camera":
        live_camera_page()
    elif page == "📤 Upload Images":
        image_upload_page()
    elif page == "📊 Dashboard":
        dashboard_page()
    elif page == "⚙️ Settings":
        settings_page()

if __name__ == "__main__":
    main()