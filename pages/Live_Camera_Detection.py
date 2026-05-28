# import streamlit as st
# import cv2
# import numpy as np
# from ultralytics import YOLO
# import pandas as pd
# import plotly.express as px
# import plotly.graph_objects as go
# from datetime import datetime
# import time
# from pathlib import Path
# from collections import defaultdict
# import io
# import base64
# from PIL import Image
# import os
# import requests

# from utils import apply_global_style
# from camera_utils import get_camera_source, open_camera, preprocess_frame

# st.set_page_config(page_title="VisionMate AI - Complete Suite", page_icon="🎯", layout="wide", initial_sidebar_state="expanded")
# apply_global_style()

# BASE_DIR = Path(__file__).parent
# possible_model_paths = [
#     BASE_DIR / "runs/detect/output/exp1/weights/best.pt",
#     BASE_DIR / "output/exp1/weights/best.pt",
#     Path("best.pt"),
# ]

# MODEL_PATH = None
# for path in possible_model_paths:
#     if path.exists():
#         MODEL_PATH = path
#         break

# CLASSES = ["aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair","cow","diningtable","dog","horse","motorbike","person","pottedplant","sheep","sofa","train","tvmonitor"]
# REAL_WIDTHS = {"person":0.5,"car":1.8,"bus":2.5,"bicycle":0.6,"motorbike":0.7,"cat":0.2,"dog":0.3,"chair":0.5,"bottle":0.1,"tvmonitor":0.9,"aeroplane":3.0,"bird":0.1,"boat":1.5,"diningtable":0.8,"horse":0.6,"sheep":0.4,"cow":0.7,"pottedplant":0.3,"sofa":0.8,"train":2.8}

# @st.cache_resource
# def load_yolo_model():
#     if MODEL_PATH is None:
#         return None
#     try:
#         model = YOLO(str(MODEL_PATH))
#         return model
#     except Exception as e:
#         st.error(f"Error loading model: {e}")
#         return None

# def estimate_distance(bbox_width_px, focal_length=700, real_width=0.5):
#     if bbox_width_px <= 0:
#         return None
#     distance = (real_width * focal_length) / bbox_width_px
#     return min(distance, 10.0)

# def analyze_with_openrouter(image, api_key, model, prompt, temperature, max_tokens):
#     try:
#         buffered = io.BytesIO()
#         image.save(buffered, format="PNG")
#         img_str = base64.b64encode(buffered.getvalue()).decode()
#         headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
#         data = {
#             "model": model,
#             "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_str}"}}]}],
#             "temperature": temperature,
#             "max_tokens": max_tokens
#         }
#         response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data, timeout=30)
#         if response.status_code == 200:
#             result = response.json()
#             return result['choices'][0]['message']['content']
#         else:
#             return f"Error: {response.status_code} - {response.text}"
#     except Exception as e:
#         return f"Error: {str(e)}"

# def export_detection_data(detection_history):
#     if not detection_history:
#         return None
#     export_data = []
#     for det in detection_history:
#         export_data.append({
#             'Timestamp': det['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
#             'Object Class': det['class'].upper(),
#             'Confidence (%)': f"{det['confidence']:.1%}",
#             'Distance (m)': f"{det['distance']:.2f}" if det['distance'] else "N/A",
#             'Bounding Box': f"{det['bbox']}"
#         })
#     df = pd.DataFrame(export_data)
#     return df.to_csv(index=False).encode('utf-8')

# def show_analytics_dashboard(detection_history, class_counts, total_detections):
#     if not detection_history:
#         st.info("No detection data available. Start camera detection to collect data.")
#         return
#     st.markdown("### 📊 Detection Analytics Dashboard")
#     col1, col2, col3, col4 = st.columns(4)
#     with col1:
#         st.markdown(f'<div class="metric-red"><h3>{total_detections}</h3><p>Total Detections</p></div>', unsafe_allow_html=True)
#     with col2:
#         unique_classes = len(class_counts)
#         st.markdown(f'<div class="metric-sky"><h3>{unique_classes}</h3><p>Unique Classes</p></div>', unsafe_allow_html=True)
#     with col3:
#         avg_confidence = sum([d['confidence'] for d in detection_history]) / len(detection_history)
#         st.markdown(f'<div class="metric-red"><h3>{avg_confidence:.1%}</h3><p>Avg Confidence</p></div>', unsafe_allow_html=True)
#     with col4:
#         total_distance = sum([d['distance'] for d in detection_history if d['distance']]) if detection_history else 0
#         st.markdown(f'<div class="metric-sky"><h3>{total_distance:.1f}m</h3><p>Total Distance</p></div>', unsafe_allow_html=True)
#     col1, col2 = st.columns(2)
#     with col1:
#         st.markdown("#### 📊 Detection by Class")
#         if class_counts:
#             df_counts = pd.DataFrame(list(class_counts.items()), columns=['Class', 'Count']).sort_values('Count', ascending=False)
#             fig = px.bar(df_counts, x='Class', y='Count', color='Count', title="Detection Frequency by Object Type", color_continuous_scale='Viridis')
#             fig.update_layout(height=400)
#             st.plotly_chart(fig, use_container_width=True)
#     with col2:
#         st.markdown("#### 📈 Confidence Distribution")
#         df_conf = pd.DataFrame([{'Confidence': d['confidence']} for d in detection_history])
#         fig = px.histogram(df_conf, x='Confidence', nbins=20, title="Confidence Score Distribution", color_discrete_sequence=['#667eea'])
#         fig.update_layout(height=400)
#         st.plotly_chart(fig, use_container_width=True)

# def main():
#     # Hardcoded OpenRouter API key (as requested)
#     OPENROUTER_API_KEY = "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"

#     st.markdown('<div class="main-header"><h1>🎯 VisionMate AI - Complete Computer Vision Suite</h1><p>Real-time Detection | AI Image Analysis | Data Export | Analytics Dashboard</p></div>', unsafe_allow_html=True)
#     yolo_model = load_yolo_model()
#     if yolo_model is None:
#         st.warning("⚠️ YOLO model not found. Camera detection will be disabled. Train the model first.")
    
#     with st.sidebar:
#         st.image("https://img.icons8.com/color/96/000000/artificial-intelligence.png", width=80)
#         st.markdown("## 🎮 Navigation")
#         page = st.radio("Select Module", ["📹 Live Camera Detection", "🖼️ AI Image Description", "📊 Analytics Dashboard", "📁 Data Export"], index=0)
#         st.markdown("---")
#         st.markdown("### ⚙️ Global Settings")
#         confidence_threshold = st.slider("Detection Confidence", 0.0, 1.0, 0.25, 0.05)
#         # Camera source selection (using your three-option helper)
#         camera_source = get_camera_source()   # returns int or string
    
#     # Initialize session state for detection data
#     if 'detection_data' not in st.session_state:
#         st.session_state.detection_data = []
#     if 'yolo_stats' not in st.session_state:
#         st.session_state.yolo_stats = {'total_detections': 0, 'unique_classes': 0, 'class_counts': {}}

#     if page == "📹 Live Camera Detection":
#         st.markdown('<div class="section-card">', unsafe_allow_html=True)
#         st.subheader("📹 Real-Time Object Detection")
#         if yolo_model is None:
#             st.error("❌ YOLO model not available. Please train the model first.")
#         else:
#             # Additional stream settings
#             process_every_n = st.slider("Process every N frames", 1, 10, 3, key="yolo_process")
#             resolution = st.selectbox("Quality Profile", ["Medium (640x480)", "Low (320x240)"], key="yolo_res")
#             width, height = (640, 480) if resolution == "Medium (640x480)" else (320, 240)
            
#             run_stream = st.checkbox("🔄 Start Video Stream", value=False, key="yolo_run")
#             status_placeholder = st.empty()
#             video_placeholder = st.empty()
#             stats_placeholder = st.empty()
#             detections_placeholder = st.empty()
            
#             def start_streaming(source, w, h, process_interval):
#                 cap = open_camera(source, w, h)
#                 if not cap or not cap.isOpened():
#                     st.error("Cannot open camera")
#                     return
                
#                 frame_count = 0
#                 total_detections = 0
#                 class_counts = defaultdict(int)
#                 detection_history = []
#                 last_time = time.time()
                
#                 try:
#                     while run_stream:
#                         ret, frame = cap.read()
#                         if not ret:
#                             time.sleep(0.05)
#                             continue
                        
#                         frame = preprocess_frame(frame, source, w, h)
#                         frame_count += 1
                        
#                         # Process every Nth frame
#                         if frame_count % process_interval == 0:
#                             results = yolo_model(frame, conf=confidence_threshold, verbose=False)
#                             detections = []
#                             annotated = frame.copy()
                            
#                             if results[0].boxes is not None:
#                                 for box in results[0].boxes:
#                                     class_name = CLASSES[int(box.cls[0])]
#                                     conf = float(box.conf[0])
#                                     bbox = box.xyxy[0].tolist()
#                                     x1, y1, x2, y2 = map(int, bbox)
#                                     bbox_width_px = x2 - x1
#                                     real_width = REAL_WIDTHS.get(class_name, 0.5)
#                                     distance = estimate_distance(bbox_width_px, real_width=real_width)
                                    
#                                     # Draw bounding box
#                                     color = (0, 255, 0)
#                                     cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
#                                     label = f"{class_name}: {conf:.1%}"
#                                     if distance:
#                                         label += f" | {distance:.1f}m"
#                                     cv2.putText(annotated, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                                    
#                                     detection = {
#                                         'class': class_name,
#                                         'confidence': conf,
#                                         'bbox': bbox,
#                                         'distance': distance,
#                                         'timestamp': datetime.now()
#                                     }
#                                     detections.append(detection)
#                                     total_detections += 1
#                                     class_counts[class_name] += 1
#                                     detection_history.append(detection)
#                                     if len(detection_history) > 500:
#                                         detection_history.pop(0)
                            
#                             # Update session state for analytics/export
#                             st.session_state.detection_data = detection_history[-30:][::-1]
#                             st.session_state.yolo_stats = {
#                                 'total_detections': total_detections,
#                                 'unique_classes': len(class_counts),
#                                 'class_counts': dict(class_counts)
#                             }
                            
#                             # Update UI status
#                             if detections:
#                                 status_placeholder.success(f"✅ Detecting {len(detections)} object(s)")
#                             else:
#                                 status_placeholder.warning("⚠️ No objects detected")
                            
#                             # Update live stats
#                             with stats_placeholder.container():
#                                 st.markdown("**Real-time Metrics**")
#                                 col_a, col_b = st.columns(2)
#                                 with col_a:
#                                     st.metric("Total Detections", total_detections)
#                                     st.metric("FPS", f"{1/(time.time()-last_time):.1f}")
#                                 with col_b:
#                                     st.metric("Unique Classes", len(class_counts))
#                                     st.metric("Current Objects", len(detections))
#                                 last_time = time.time()
                            
#                             with detections_placeholder.container():
#                                 if detections:
#                                     st.markdown("**Current Detections:**")
#                                     for det in detections[:5]:
#                                         dist_text = f"{det['distance']:.1f}m" if det['distance'] else "N/A"
#                                         st.markdown(f"• **{det['class'].upper()}** - {det['confidence']:.1%} - 📏 {dist_text}")
                            
#                             frame_to_show = annotated
#                         else:
#                             frame_to_show = frame
                        
#                         # Display frame
#                         frame_rgb = cv2.cvtColor(frame_to_show, cv2.COLOR_BGR2RGB)
#                         video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
#                         time.sleep(0.01)
                        
#                 except Exception as e:
#                     st.error(f"Stream error: {e}")
#                 finally:
#                     cap.release()
#                     video_placeholder.empty()
#                     status_placeholder.empty()
#                     stats_placeholder.empty()
#                     detections_placeholder.empty()
            
#             if run_stream:
#                 # Convert camera_source (which may be int or string) to the actual source value
#                 src = camera_source
#                 start_streaming(src, width, height, process_every_n)
#             else:
#                 st.write("Stopped. Check the box above to start streaming.")
#         st.markdown('</div>', unsafe_allow_html=True)

#     elif page == "🖼️ AI Image Description":
#         st.markdown('<div class="section-card">', unsafe_allow_html=True)
#         st.subheader("🖼️ AI-Powered Image Description")
#         col1, col2 = st.columns([1, 1])
#         result = "Please upload the image to run the analysis."
#         with col1:
#             st.markdown("#### 📤 Upload Image")
#             uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png', 'webp'])
#             if uploaded_file is not None:
#                 image = Image.open(uploaded_file)
#                 st.image(image, caption="Uploaded Image", use_container_width=True)
#                 st.caption(f"Size: {image.size[0]} x {image.size[1]} pixels")
#         with col2:
#             st.markdown("#### 🤖 AI Analysis Settings")
#             model = st.selectbox("AI Model", ["meta-llama/llama-4-scout"])
#             description_style = st.selectbox("Description Style", ["Detailed Description", "Brief Description", "Technical Analysis", "Artistic Description", "Accessibility (Alt Text)"])
#             temperature = st.slider("Creativity", 0.0, 1.0, 0.7, 0.1)
#             if uploaded_file is not None and st.button("🔍 Analyze Image", type="primary", use_container_width=True):
#                 with st.spinner("🧠 AI is analyzing your image..."):
#                     prompts = {
#                         "Detailed Description": "Provide a detailed, comprehensive description of this image. Include main subjects, colors, composition, mood, and notable details.",
#                         "Brief Description": "Provide a brief, concise description (2-3 sentences) of what's in this image.",
#                         "Technical Analysis": "Analyze this image technically: composition, lighting, color palette, depth of field, and photographic elements.",
#                         "Artistic Description": "Describe this image from an artistic perspective: mood, atmosphere, emotional impact, and visual storytelling.",
#                         "Accessibility (Alt Text)": "Write clear, descriptive alt text for this image suitable for accessibility purposes."
#                     }
#                     prompt = prompts.get(description_style, prompts["Detailed Description"])
#                     result = analyze_with_openrouter(image, OPENROUTER_API_KEY, model, prompt, temperature, max_tokens=500)
#         st.markdown("#### 📝 Analysis Result")
#         st.markdown(f'<div style="background: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 4px solid #667eea;">{result}</div>', unsafe_allow_html=True)
#         st.markdown('</div>', unsafe_allow_html=True)

#     elif page == "📊 Analytics Dashboard":
#         st.markdown('<div class="section-card">', unsafe_allow_html=True)
#         st.subheader("📊 Detection Analytics Dashboard")
#         if st.session_state.detection_data:
#             detection_history = st.session_state.detection_data
#             class_counts = defaultdict(int)
#             for det in detection_history:
#                 class_counts[det['class']] += 1
#             show_analytics_dashboard(detection_history, class_counts, len(detection_history))
#         else:
#             st.info("📭 No detection data available. Please run the camera detection first to collect data.")
#             st.markdown("### How to get data:\n1. Go to **Live Camera Detection** page\n2. Click **Start Camera**\n3. Let the camera detect objects for a few seconds\n4. Return here to see analytics")
#         st.markdown('</div>', unsafe_allow_html=True)

#     elif page == "📁 Data Export":
#         st.markdown('<div class="section-card">', unsafe_allow_html=True)
#         st.subheader("📁 Export Detection Data")
#         if st.session_state.detection_data:
#             detection_history = st.session_state.detection_data
#             st.markdown("#### 📋 Data Preview")
#             preview_data = []
#             for det in detection_history[:10]:
#                 preview_data.append({'Time': det['timestamp'].strftime('%H:%M:%S'), 'Object': det['class'].upper(), 'Confidence': f"{det['confidence']:.1%}", 'Distance': f"{det['distance']:.2f}m" if det['distance'] else "N/A"})
#             st.dataframe(pd.DataFrame(preview_data), use_container_width=True)
#             st.markdown("#### 💾 Export Options")
#             col1, col2, col3 = st.columns(3)
#             with col1:
#                 csv_data = export_detection_data(detection_history)
#                 if csv_data:
#                     st.download_button(label="📥 Download CSV", data=csv_data, file_name=f"detection_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv", use_container_width=True)
#             with col2:
#                 summary = {'Total Detections': len(detection_history), 'Unique Classes': len(set([d['class'] for d in detection_history])), 'Average Confidence': f"{sum([d['confidence'] for d in detection_history]) / len(detection_history):.1%}", 'Most Detected': max(set([d['class'] for d in detection_history]), key=lambda x: sum(1 for d in detection_history if d['class'] == x)) if detection_history else "None"}
#                 st.json(summary)
#             with col3:
#                 if st.button("🗑️ Clear All Data", use_container_width=True):
#                     st.session_state.detection_data = []
#                     st.session_state.yolo_stats = {'total_detections': 0, 'unique_classes': 0, 'class_counts': {}}
#                     st.rerun()
#             st.markdown("#### 📊 Detailed Statistics by Class")
#             class_stats = defaultdict(lambda: {'count': 0, 'total_conf': 0, 'total_dist': 0})
#             for det in detection_history:
#                 class_stats[det['class']]['count'] += 1
#                 class_stats[det['class']]['total_conf'] += det['confidence']
#                 if det['distance']:
#                     class_stats[det['class']]['total_dist'] += det['distance']
#             stats_data = []
#             for class_name, stats in class_stats.items():
#                 stats_data.append({'Class': class_name.upper(), 'Detections': stats['count'], 'Avg Confidence': f"{stats['total_conf'] / stats['count']:.1%}", 'Avg Distance': f"{stats['total_dist'] / stats['count']:.2f}m" if stats['total_dist'] > 0 else "N/A", 'Percentage': f"{(stats['count'] / len(detection_history) * 100):.1f}%"})
#             st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)
#         else:
#             st.info("📭 No data available to export. Please run camera detection first.")
#         st.markdown('</div>', unsafe_allow_html=True)

#     st.markdown("---")
#     st.markdown('<div style="text-align: center; color: #666; font-size: 12px;">VisionMate AI Suite | Powered by YOLOv8 + OpenRouter | Real-time Detection | AI Analysis</div>', unsafe_allow_html=True)

# if __name__ == "__main__":
#     main()
import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
from pathlib import Path
from collections import defaultdict
import io
import base64
from PIL import Image
import os
import requests

from camera_utils import get_camera_source, open_camera, preprocess_frame, release_camera
from utils import apply_global_style

# ============================================
# PAGE CONFIG & SOFT DARK THEME (Attractive, readable)
# ============================================
st.set_page_config(
    page_title="VisionMate AI - Complete Suite",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_global_style()

# ============================================
# CONFIGURATION (unchanged)
# ============================================
BASE_DIR = Path(__file__).parent
possible_model_paths = [
    BASE_DIR / "runs/detect/output/exp1/weights/best.pt",
    BASE_DIR / "output/exp1/weights/best.pt",
    Path("best.pt"),
]

MODEL_PATH = None
for path in possible_model_paths:
    if path.exists():
        MODEL_PATH = path
        break

CLASSES = ["aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair","cow",
           "diningtable","dog","horse","motorbike","person","pottedplant","sheep","sofa","train","tvmonitor"]
REAL_WIDTHS = {"person":0.5,"car":1.8,"bus":2.5,"bicycle":0.6,"motorbike":0.7,"cat":0.2,"dog":0.3,
               "chair":0.5,"bottle":0.1,"tvmonitor":0.9,"aeroplane":3.0,"bird":0.1,"boat":1.5,
               "diningtable":0.8,"horse":0.6,"sheep":0.4,"cow":0.7,"pottedplant":0.3,"sofa":0.8,"train":2.8}

OPENROUTER_API_KEY = "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"

# ============================================
# HELPER FUNCTIONS (unchanged)
# ============================================
@st.cache_resource
def load_yolo_model():
    if MODEL_PATH is None:
        return None
    try:
        model = YOLO(str(MODEL_PATH))
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def estimate_distance(bbox_width_px, focal_length=700, real_width=0.5):
    if bbox_width_px <= 0:
        return None
    distance = (real_width * focal_length) / bbox_width_px
    return min(distance, 10.0)

def analyze_with_openrouter(image, api_key, model, prompt, temperature, max_tokens):
    try:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        data = {
            "model": model,
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt},
                       {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_str}"}}]}],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data, timeout=30)
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            return f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Error: {str(e)}"

def export_detection_data(detection_history):
    if not detection_history:
        return None
    export_data = []
    for det in detection_history:
        export_data.append({
            'Timestamp': det['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
            'Object Class': det['class'].upper(),
            'Confidence (%)': f"{det['confidence']:.1%}",
            'Distance (m)': f"{det['distance']:.2f}" if det['distance'] else "N/A",
            'Bounding Box': f"{det['bbox']}"
        })
    df = pd.DataFrame(export_data)
    return df.to_csv(index=False).encode('utf-8')

def show_analytics_dashboard(detection_history, class_counts, total_detections):
    if not detection_history:
        st.info("No detection data available. Start camera detection to collect data.")
        return
    st.markdown("### 📊 Detection Analytics Dashboard")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-red"><h3>{total_detections}</h3><p>Total Detections</p></div>', unsafe_allow_html=True)
    with col2:
        unique_classes = len(class_counts)
        st.markdown(f'<div class="metric-sky"><h3>{unique_classes}</h3><p>Unique Classes</p></div>', unsafe_allow_html=True)
    with col3:
        avg_confidence = sum([d['confidence'] for d in detection_history]) / len(detection_history)
        st.markdown(f'<div class="metric-red"><h3>{avg_confidence:.1%}</h3><p>Avg Confidence</p></div>', unsafe_allow_html=True)
    with col4:
        total_distance = sum([d['distance'] for d in detection_history if d['distance']]) if detection_history else 0
        st.markdown(f'<div class="metric-sky"><h3>{total_distance:.1f}m</h3><p>Total Distance</p></div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 📊 Detection by Class")
        if class_counts:
            df_counts = pd.DataFrame(list(class_counts.items()), columns=['Class', 'Count']).sort_values('Count', ascending=False)
            fig = px.bar(df_counts, x='Class', y='Count', color='Count', title="Detection Frequency by Object Type", color_continuous_scale='Viridis')
            fig.update_layout(height=400, paper_bgcolor='#23263b', font_color='#eef2ff', plot_bgcolor='#1a1c2e')
            st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown("#### 📈 Confidence Distribution")
        df_conf = pd.DataFrame([{'Confidence': d['confidence']} for d in detection_history])
        fig = px.histogram(df_conf, x='Confidence', nbins=20, title="Confidence Score Distribution", color_discrete_sequence=['#ff4444'])
        fig.update_layout(height=400, paper_bgcolor='#23263b', font_color='#eef2ff', plot_bgcolor='#1a1c2e')
        st.plotly_chart(fig, use_container_width=True)

# ============================================
# MAIN APP (identical logic, only UI theme changed)
# ============================================
def main():
    st.markdown('<div class="main-header"><h1>🎯 VisionMate AI - Complete Computer Vision Suite</h1><p>Real-time Detection | AI Image Analysis | Data Export | Analytics Dashboard</p></div>', unsafe_allow_html=True)
    
    yolo_model = load_yolo_model()
    if yolo_model is None:
        st.warning("⚠️ YOLO model not found. Camera detection will be disabled. Train the model first.")
    
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/artificial-intelligence.png", width=80)
        st.markdown("## 🎮 Navigation")
        page = st.radio("Select Module", ["📹 Live Camera Detection", "🖼️ AI Image Description", "📊 Analytics Dashboard", "📁 Data Export"], index=0)
        st.markdown("---")
        st.markdown("### ⚙️ Global Settings")
        confidence_threshold = st.slider("Detection Confidence", 0.0, 1.0, 0.25, 0.05)
        camera_source = get_camera_source()
    
    if 'detection_data' not in st.session_state:
        st.session_state.detection_data = []
    if 'yolo_stats' not in st.session_state:
        st.session_state.yolo_stats = {'total_detections': 0, 'unique_classes': 0, 'class_counts': {}}
    
    # PAGE 1: LIVE CAMERA DETECTION
    if page == "📹 Live Camera Detection":
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("📹 Real-Time Object Detection")
        if yolo_model is None:
            st.error("❌ YOLO model not available. Please train the model first.")
        else:
            col1, col2, col3 = st.columns(3)
            with col1:
                process_every_n = st.slider("Process every N frames", 1, 10, 3, key="yolo_process")
            with col2:
                resolution = st.selectbox("Quality Profile", ["Medium (640x480)", "Low (320x240)"], key="yolo_res")
                width, height = (640, 480) if resolution == "Medium (640x480)" else (320, 240)
            with col3:
                run_stream = st.checkbox("🔄 START CAMERA", value=False, key="yolo_run")
            
            status_placeholder = st.empty()
            video_placeholder = st.empty()
            stats_placeholder = st.empty()
            detections_placeholder = st.empty()
            
            def start_streaming(source, w, h, process_interval):
                cap = open_camera(source, w, h)
                if cap is None:
                    st.error(f"Cannot open camera source: {source}. Try a different option.")
                    return
                
                frame_count = 0
                total_detections = 0
                class_counts = defaultdict(int)
                detection_history = []
                last_time = time.time()
                
                try:
                    while run_stream:
                        ret, frame = cap.read()
                        if not ret:
                            time.sleep(0.02)
                            continue
                        frame = preprocess_frame(frame, source, w, h)
                        frame_count += 1
                        
                        if frame_count % process_interval == 0:
                            results = yolo_model(frame, conf=confidence_threshold, verbose=False)
                            detections = []
                            annotated = frame.copy()
                            
                            if results[0].boxes is not None:
                                for box in results[0].boxes:
                                    class_name = CLASSES[int(box.cls[0])]
                                    conf = float(box.conf[0])
                                    bbox = box.xyxy[0].tolist()
                                    x1, y1, x2, y2 = map(int, bbox)
                                    bbox_width_px = x2 - x1
                                    real_width = REAL_WIDTHS.get(class_name, 0.5)
                                    distance = estimate_distance(bbox_width_px, real_width=real_width)
                                    
                                    color = (0, 255, 0)
                                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                                    label = f"{class_name}: {conf:.1%}"
                                    if distance:
                                        label += f" | {distance:.1f}m"
                                    cv2.putText(annotated, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                                    
                                    detection = {
                                        'class': class_name,
                                        'confidence': conf,
                                        'bbox': bbox,
                                        'distance': distance,
                                        'timestamp': datetime.now()
                                    }
                                    detections.append(detection)
                                    total_detections += 1
                                    class_counts[class_name] += 1
                                    detection_history.append(detection)
                                    if len(detection_history) > 500:
                                        detection_history.pop(0)
                            
                            st.session_state.detection_data = detection_history[-30:][::-1]
                            st.session_state.yolo_stats = {
                                'total_detections': total_detections,
                                'unique_classes': len(class_counts),
                                'class_counts': dict(class_counts)
                            }
                            
                            if detections:
                                status_placeholder.success(f"✅ Detecting {len(detections)} object(s)")
                            else:
                                status_placeholder.warning("⚠️ No objects detected")
                            
                            with stats_placeholder.container():
                                col_a, col_b = st.columns(2)
                                with col_a:
                                    st.metric("Total Detections", total_detections)
                                    st.metric("FPS", f"{1/(time.time()-last_time):.1f}")
                                with col_b:
                                    st.metric("Unique Classes", len(class_counts))
                                    st.metric("Current Objects", len(detections))
                                last_time = time.time()
                            
                            with detections_placeholder.container():
                                if detections:
                                    st.markdown("**🔍 Current Detections:**")
                                    for det in detections[:5]:
                                        dist_text = f"{det['distance']:.1f}m" if det['distance'] else "N/A"
                                        st.markdown(f'<div class="detection-item">• <b>{det["class"].upper()}</b> - {det["confidence"]:.1%} - 📏 {dist_text}</div>', unsafe_allow_html=True)
                            
                            frame_to_show = annotated
                        else:
                            frame_to_show = frame
                        
                        frame_rgb = cv2.cvtColor(frame_to_show, cv2.COLOR_BGR2RGB)
                        video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                        
                except Exception as e:
                    st.error(f"Stream error: {e}")
                finally:
                    release_camera(cap)
                    video_placeholder.empty()
                    status_placeholder.empty()
                    stats_placeholder.empty()
                    detections_placeholder.empty()
            
            if run_stream:
                start_streaming(camera_source, width, height, process_every_n)
            else:
                st.info("Press **START CAMERA** to begin real-time object detection.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # PAGE 2: AI IMAGE DESCRIPTION
    elif page == "🖼️ AI Image Description":
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("🖼️ AI-Powered Image Description")
        col1, col2 = st.columns([1, 1])
        result = "Please upload the image to run the analysis."
        with col1:
            st.markdown("#### 📤 Upload Image")
            uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png', 'webp'])
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_container_width=True)
                st.caption(f"Size: {image.size[0]} x {image.size[1]} pixels")
        with col2:
            st.markdown("#### 🤖 AI Analysis Settings")
            model = st.selectbox("AI Model", ["meta-llama/llama-4-scout"])
            description_style = st.selectbox("Description Style", ["Detailed Description", "Brief Description", "Technical Analysis", "Artistic Description", "Accessibility (Alt Text)"])
            temperature = st.slider("Creativity", 0.0, 1.0, 0.7, 0.1)
            if uploaded_file is not None and st.button("🔍 Analyze Image", type="primary", use_container_width=True):
                with st.spinner("🧠 AI is analyzing your image..."):
                    prompts = {
                        "Detailed Description": "Provide a detailed, comprehensive description of this image. Include main subjects, colors, composition, mood, and notable details.",
                        "Brief Description": "Provide a brief, concise description (2-3 sentences) of what's in this image.",
                        "Technical Analysis": "Analyze this image technically: composition, lighting, color palette, depth of field, and photographic elements.",
                        "Artistic Description": "Describe this image from an artistic perspective: mood, atmosphere, emotional impact, and visual storytelling.",
                        "Accessibility (Alt Text)": "Write clear, descriptive alt text for this image suitable for accessibility purposes."
                    }
                    prompt = prompts.get(description_style, prompts["Detailed Description"])
                    result = analyze_with_openrouter(image, OPENROUTER_API_KEY, model, prompt, temperature, max_tokens=500)
        st.markdown("#### 📝 Analysis Result")
        st.markdown(f'<div style="background: #1e2137; padding: 20px; border-radius: 12px; border-left: 4px solid #ff4444; color: #eef2ff;">{result}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # PAGE 3: ANALYTICS DASHBOARD
    elif page == "📊 Analytics Dashboard":
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("📊 Detection Analytics Dashboard")
        if st.session_state.detection_data:
            detection_history = st.session_state.detection_data
            class_counts = defaultdict(int)
            for det in detection_history:
                class_counts[det['class']] += 1
            show_analytics_dashboard(detection_history, class_counts, len(detection_history))
        else:
            st.info("📭 No detection data available. Please run the camera detection first to collect data.")
            st.markdown("### How to get data:\n1. Go to **Live Camera Detection** page\n2. Click **Start Camera**\n3. Let the camera detect objects for a few seconds\n4. Return here to see analytics")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # PAGE 4: DATA EXPORT
    elif page == "📁 Data Export":
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("📁 Export Detection Data")
        if st.session_state.detection_data:
            detection_history = st.session_state.detection_data
            st.markdown("#### 📋 Data Preview")
            preview_data = []
            for det in detection_history[:10]:
                preview_data.append({'Time': det['timestamp'].strftime('%H:%M:%S'), 'Object': det['class'].upper(), 'Confidence': f"{det['confidence']:.1%}", 'Distance': f"{det['distance']:.2f}m" if det['distance'] else "N/A"})
            st.dataframe(pd.DataFrame(preview_data), use_container_width=True)
            st.markdown("#### 💾 Export Options")
            col1, col2, col3 = st.columns(3)
            with col1:
                csv_data = export_detection_data(detection_history)
                if csv_data:
                    st.download_button(label="📥 Download CSV", data=csv_data, file_name=f"detection_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv", use_container_width=True)
            with col2:
                summary = {'Total Detections': len(detection_history), 'Unique Classes': len(set([d['class'] for d in detection_history])), 'Average Confidence': f"{sum([d['confidence'] for d in detection_history]) / len(detection_history):.1%}", 'Most Detected': max(set([d['class'] for d in detection_history]), key=lambda x: sum(1 for d in detection_history if d['class'] == x)) if detection_history else "None"}
                st.json(summary)
            with col3:
                if st.button("🗑️ Clear All Data", use_container_width=True):
                    st.session_state.detection_data = []
                    st.session_state.yolo_stats = {'total_detections': 0, 'unique_classes': 0, 'class_counts': {}}
                    st.rerun()
            st.markdown("#### 📊 Detailed Statistics by Class")
            class_stats = defaultdict(lambda: {'count': 0, 'total_conf': 0, 'total_dist': 0})
            for det in detection_history:
                class_stats[det['class']]['count'] += 1
                class_stats[det['class']]['total_conf'] += det['confidence']
                if det['distance']:
                    class_stats[det['class']]['total_dist'] += det['distance']
            stats_data = []
            for class_name, stats in class_stats.items():
                stats_data.append({'Class': class_name.upper(), 'Detections': stats['count'], 'Avg Confidence': f"{stats['total_conf'] / stats['count']:.1%}", 'Avg Distance': f"{stats['total_dist'] / stats['count']:.2f}m" if stats['total_dist'] > 0 else "N/A", 'Percentage': f"{(stats['count'] / len(detection_history) * 100):.1f}%"})
            st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)
        else:
            st.info("📭 No data available to export. Please run camera detection first.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown('<div style="text-align: center; color: #888; font-size: 12px;">VisionMate AI Suite | Powered by YOLOv8 + OpenRouter | Real-time Detection | AI Analysis</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
