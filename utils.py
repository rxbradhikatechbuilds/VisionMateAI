# import streamlit as st
# import numpy as np
# from PIL import Image, ImageDraw, ImageFont
# import random
# import hashlib
# import time

# def apply_global_style():
#     st.markdown("""
#     <style>
#     .stApp {
#         background: linear-gradient(135deg, #f8fafc 0%, #eef2f5 100%);
#     }
#     [data-testid="stSidebar"] {
#         background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
#         border-right: 2px solid #e74c3c;
#     }
#     [data-testid="stSidebar"] * {
#         color: #f1f5f9 !important;
#     }
#     .stButton > button {
#         background: linear-gradient(90deg, #e74c3c 0%, #c0392b 100%);
#         color: white !important;
#         border: none;
#         border-radius: 12px;
#         font-weight: 600;
#         transition: 0.3s;
#     }
#     .stButton > button:hover {
#         transform: translateY(-2px);
#         box-shadow: 0 6px 12px rgba(231,76,60,0.3);
#     }
#     h1, h2, h3 {
#         color: #1e293b !important;
#         font-weight: 700;
#     }
#     .hero-section {
#         text-align: center;
#         padding: 2rem;
#         background: linear-gradient(120deg, #e74c3c, #3b82f6);
#         border-radius: 28px;
#         margin-bottom: 2rem;
#         color: white;
#         box-shadow: 0 10px 25px rgba(0,0,0,0.1);
#     }
#     .feature-card {
#         background: white;
#         border-radius: 20px;
#         padding: 1.5rem;
#         margin: 1rem 0;
#         border-left: 6px solid #3b82f6;
#         box-shadow: 0 4px 12px rgba(0,0,0,0.05);
#     }
#     .feature-item {
#         background: white;
#         padding: 1rem;
#         border-radius: 14px;
#         margin: 0.5rem 0;
#         border: 1px solid #e2e8f0;
#         transition: 0.2s;
#     }
#     .feature-item:hover {
#         border-color: #e74c3c;
#         transform: scale(1.01);
#     }
#     .feature-title {
#         font-weight: 700;
#         color: #e74c3c;
#         margin-bottom: 0.3rem;
#     }
#     .metric-red {
#         background: linear-gradient(135deg, #e74c3c, #c0392b);
#         padding: 1rem;
#         border-radius: 16px;
#         color: white;
#         text-align: center;
#     }
#     .metric-sky {
#         background: linear-gradient(135deg, #3b82f6, #2563eb);
#         padding: 1rem;
#         border-radius: 16px;
#         color: white;
#         text-align: center;
#     }
#     </style>
#     """, unsafe_allow_html=True)

# def simulate_detection(image, module, conf_threshold=0.5):
#     img_hash = hashlib.md5(image.tobytes()).hexdigest()[:8]
#     random.seed(int(img_hash, 16) % 10**6)
#     draw = ImageDraw.Draw(image)
#     width, height = image.size
#     try:
#         font = ImageFont.truetype("arial.ttf", 18)
#     except:
#         font = ImageFont.load_default()
#     detections = []
#     if module == "fire":
#         possible_labels = ["🔥 fire", "💨 smoke", "🔥 flame"]
#         colors = ["#FF4500", "#B22222", "#DC143C"]
#         num_boxes = random.randint(1, 3)
#     elif module == "gesture":
#         possible_labels = ["👍 thumbs_up", "✌️ victory", "✋ open_palm", "👊 fist", "☝️ pointing"]
#         colors = ["#1E90FF", "#32CD32", "#FFA500", "#9932CC"]
#         num_boxes = random.randint(1, 2)
#     else:
#         possible_labels = ["person", "car", "laptop", "phone", "book", "bottle"]
#         colors = ["#2E8B57", "#4169E1", "#FF8C00", "#6A5ACD"]
#         num_boxes = random.randint(2, 5)
#     for _ in range(num_boxes):
#         x1 = random.randint(int(width*0.1), int(width*0.7))
#         y1 = random.randint(int(height*0.1), int(height*0.6))
#         x2 = x1 + random.randint(int(width*0.1), int(width*0.3))
#         y2 = y1 + random.randint(int(height*0.1), int(height*0.3))
#         x2 = min(x2, width-10)
#         y2 = min(y2, height-10)
#         conf = round(random.uniform(conf_threshold, 0.98), 2)
#         label = random.choice(possible_labels)
#         color = random.choice(colors)
#         draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
#         draw.rectangle([x1+2, y1+2, x2-2, y2-2], outline=color, width=1)
#         text = f"{label} {conf:.2f}"
#         bbox = draw.textbbox((x1, y1-20), text, font=font)
#         draw.rectangle(bbox, fill=color)
#         draw.text((x1, y1-20), text, fill="white", font=font)
#         detections.append({"label": label, "confidence": conf, "box": (x1, y1, x2, y2)})
#     return image, detections

# def detection_ui(module_key, module_name, upload_text, button_text):
#     col_left, col_right = st.columns([1, 1], gap="large")
#     with col_left:
#         uploaded_file = st.file_uploader(upload_text, type=["png", "jpg", "jpeg"], key=f"upload_{module_key}")
#         conf = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01, key=f"conf_{module_key}")
#         if st.button(button_text, key=f"btn_{module_key}"):
#             if uploaded_file:
#                 with st.spinner("Analyzing..."):
#                     time.sleep(0.5)
#                     image = Image.open(uploaded_file).convert("RGB")
#                     annotated, dets = simulate_detection(image, module_key, conf)
#                     st.session_state[f"detection_result_{module_key}"] = (annotated, dets)
#                     st.success(f"✅ Found {len(dets)} objects")
#             else:
#                 st.warning("Please upload an image first.")
#     with col_right:
#         st.markdown(f"#### 🔍 {module_name} Output")
#         result = st.session_state.get(f"detection_result_{module_key}")
#         if result:
#             annotated_img, detections = result
#             st.image(annotated_img, use_container_width=True)
#             if detections:
#                 avg_conf = np.mean([d['confidence'] for d in detections])
#                 st.metric("Average Confidence", f"{avg_conf*100:.1f}%")
#                 for d in detections:
#                     st.caption(f"• {d['label']} – {d['confidence']*100:.1f}%")
#                 st.metric("Total Objects", len(detections))
#         else:
#             st.info("Upload an image and click 'Detect' to see results.")

import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random
import hashlib
import time

def apply_global_style():
    """
    Applies a clean 3-color rule style framework that honors Streamlit's 
    Light/Dark/System theme choices perfectly.
    """
    st.markdown("""
    <style>
    /* Global Variables dynamically aligned with Streamlit Theme Engine */
    :root {
        --accent-color: #4f46e5;
        --accent-hover: #4338ca;
        --card-border: rgba(128, 128, 128, 0.15);
    }

    /* 60% Dominant: Seamless structural backgrounds */
    .stApp {
        background-color: transparent !important;
    }
    
    /* 30% Secondary: Sidebar and standard structural layout spacing */
    [data-testid="stSidebar"] {
        border-right: 1px solid var(--card-border);
    }
    
    /* 10% Accent: Standard interactive components */
    .stButton > button {
        background: var(--accent-color) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.5rem 1rem !important;
        transition: all 0.25s ease-in-out !important;
    }
    
    .stButton > button:hover {
        background: var(--accent-hover) !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3);
    }

    /* Clean typography structure mapping to system/native values */
    h1, h2, h3, h4, h5, h6 {
        font-weight: 700 !important;
    }
    
    /* Custom Components built around the 3-color scheme rules */
    .hero-section {
        text-align: center;
        padding: 2.5rem 1.5rem;
        background: linear-gradient(135deg, var(--accent-color) 0%, #312e81 100%);
        border-radius: 16px;
        margin-bottom: 2rem;
        color: #ffffff !important;
        box-shadow: 0 10px 25px rgba(0,0,0,0.05);
    }
    .hero-section h1, .hero-section p {
        color: #ffffff !important;
    }

    .feature-card {
        background-color: rgba(128, 128, 128, 0.06);
        border: 1px solid var(--card-border);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 5px solid var(--accent-color);
    }

    .feature-item {
        background-color: rgba(128, 128, 128, 0.04);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border: 1px solid var(--card-border);
        transition: all 0.2s ease;
    }
    .feature-item:hover {
        border-color: var(--accent-color);
        transform: scale(1.005);
    }
    .feature-title {
        font-weight: 700;
        color: var(--accent-color) !important;
        margin-bottom: 0.3rem;
    }

    /* Standardized Metric boxes matching 30% background/10% element accents */
    .metric-red, .metric-sky {
        background-color: rgba(128, 128, 128, 0.06);
        border: 1px solid var(--card-border);
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-red h3, .metric-sky h3 {
        color: var(--accent-color) !important;
        font-size: 2rem;
        margin: 0;
    }
    .metric-red p, .metric-sky p {
        margin: 0;
        font-size: 0.9rem;
        opacity: 0.85;
    }
    </style>
    """, unsafe_allow_html=True)

def simulate_detection(image, module, conf_threshold=0.5):
    img_hash = hashlib.md5(image.tobytes()).hexdigest()[:8]
    random.seed(int(img_hash, 16) % 10**6)
    draw = ImageDraw.Draw(image)
    width, height = image.size
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except:
        font = ImageFont.load_default()
    
    detections = []
    if module == "fire":
        possible_labels = ["🔥 fire", "💨 smoke", "🔥 flame"]
        colors = ["#FF4500", "#B22222", "#DC143C"]
        num_boxes = random.randint(1, 3)
    elif module == "gesture":
        possible_labels = ["👍 thumbs_up", "✌️ victory", "✋ open_palm", "👊 fist"]
        colors = ["#4F46E5", "#10B981", "#F59E0B", "#EF4444"]
        num_boxes = random.randint(1, 2)
        
    return image, detections