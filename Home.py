import streamlit as st
from utils import apply_global_style

st.set_page_config(
    page_title="VisionAI – Smart Computer Vision Suite",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_global_style()

# Hero Section
st.markdown("""
    <div class="hero-section">
        <h1>👁️ VisionMateAI</h1>
        <p style="font-size: 1.3rem;">Real‑time Intelligence · Edge AI · Multimodal Vision</p>
    </div>
""", unsafe_allow_html=True)

# Problem & Solution Cards
col1, col2 = st.columns(2)
with col1:
    st.markdown("""
        <div class="feature-card">
            <h3>❌ The Challenge</h3>
            <p style="text-align: justify; margin-bottom: 0;">Manual inspection of images and video is slow, error‑prone, and unable to scale. 
            Fires go undetected, gestures unrecognized, objects uncounted.</p>
        </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
        <div class="feature-card" style="border-left-color: #e74c3c;">
            <h3>✅ Our Solution</h3>
            <p style="text-align: justify; margin-bottom: 0;"><strong>VisionAI</strong> combines YOLOv8, MediaPipe, FastAPI, and Large Vision Models 
            into one seamless interface – delivering instant, actionable insights.</p>
        </div>
    """, unsafe_allow_html=True)

st.divider()

# Feature Grid (10 core capabilities)
st.markdown("### ⚡ Key Capabilities")

features = [
    ("1. Selected Object Counting", "Filter & track specific object types in live feeds."),
    ("2. Scene Detection", "Identify environment type and spatial layout."),
    ("3. Automated Insights", "Correlate object volume and processing trends."),
    ("4. Action Suggestions", "Context‑aware advice based on hazard states."),
    ("5. Distance Estimation", "Geometric bounding‑box distance (up to 10m)."),
    ("6. Detailed Scene Description", "Natural‑language summaries via OpenRouter LLMs."),
    ("7. Dynamic Safety Alerts", "Immediate alerts for fire, breaches, or anomalies."),
    ("8. Study Assistant", "Track books / objects & detect distractions."),
    ("9. Retail Analytics", "Item concentration, frequency, unique counts."),
    ("10. Basic Gesture Tracking", "Thumbs‑up, victory, open palm, fist, pointing.")
]

cols = st.columns(2)
for i, (title, desc) in enumerate(features):
    with cols[i % 2]:
        st.markdown(f"""
            <div class="feature-item">
                <div class="feature-title">{title}</div>
                <div style="opacity: 0.9;">{desc}</div>
            </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.info("👈 **Use the sidebar to launch any detection module** – no separate setup required for the core features.")