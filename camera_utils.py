
# """
# camera_utils.py – Optimized for all three apps (Gesture, Fire, VisionMate)
# Supports:
# - Laptop integrated camera (index 0)
# - External USB webcam (index 1, with CAP_DSHOW on Windows)
# - Mobile IP camera (DroidCam, IP Webcam) via URL
# """

# import cv2
# import streamlit as st
# import os

# def get_camera_source():
#     """Display sidebar widgets to select camera source."""
#     st.sidebar.markdown("### 📷 Camera Source")
#     camera_choice = st.sidebar.selectbox(
#         "Select Camera",
#         ("Laptop Integrated Camera", "External USB Webcam", "Mobile Phone (via DroidCam)")
#     )
#     mobile_ip = st.sidebar.text_input("Mobile IP (for DroidCam)", value="192.168.1.50")
    
#     if camera_choice == "Laptop Integrated Camera":
#         return 0
#     elif camera_choice == "External USB Webcam":
#         return 1
#     else:
#         # DroidCam HTTP stream URL (default port 4747)
#         return f"http://192.168.1.22:4747/video"

# def open_camera(source, width=320, height=240):
#     """
#     Open camera source with proper backend.
#     Returns cv2.VideoCapture object or None if failed.
#     """
#     if isinstance(source, str):
#         # IP camera stream (e.g., DroidCam)
#         cap = cv2.VideoCapture(source)
#     else:
#         # Integer camera index (0,1,2...)
#         if os.name == 'nt':  # Windows
#             cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
#         else:                # Linux / macOS
#             cap = cv2.VideoCapture(source)
#         # Set resolution for local cameras
#         cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
#         cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
#     if not cap.isOpened():
#         return None
#     return cap

# def preprocess_frame(frame, source, target_width, target_height, rotate=False):
#     """
#     Preprocess frame: resize for IP cameras, optionally rotate.
#     """
#     if isinstance(source, str):
#         # IP camera: resize to target dimensions
#         frame = cv2.resize(frame, (target_width, target_height))
#         if rotate:
#             frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
#     # For local cameras, frame is already at target resolution
#     return frame

# def release_camera(cap):
#     """Safely release the camera capture object."""
#     if cap is not None:
#         cap.release()

"""
camera_utils.py – Universal camera module that works on:
- Local machine (Windows/Linux/macOS) with OpenCV real‑time video
- Streamlit Cloud (no physical camera) using st.camera_input (browser camera)

Usage in your main app:
    import camera_utils as cam

    def process_frame(frame):
        # Your gesture/fire/vision processing here
        return frame   # or None to skip displaying

    cam.run_camera_loop(process_frame)
"""

import cv2
import streamlit as st
import os
import time
import numpy as np

# ----------------------------------------------------------------------
# Environment detection
# ----------------------------------------------------------------------
def is_cloud():
    """Return True if running on Streamlit Cloud."""
    return os.environ.get('STREAMLIT_SERVER_PORT') is not None


# ----------------------------------------------------------------------
# Local OpenCV camera functions (used when running locally)
# ----------------------------------------------------------------------
def get_camera_source():
    """Display sidebar widgets to select camera source (local mode only)."""
    st.sidebar.markdown("### 📷 Camera Source")
    camera_choice = st.sidebar.selectbox(
        "Select Camera",
        ("Laptop Integrated Camera", "External USB Webcam", "Mobile Phone (via DroidCam)")
    )
    mobile_ip = st.sidebar.text_input("Mobile IP (for DroidCam)", value="192.168.1.50")

    if camera_choice == "Laptop Integrated Camera":
        return 0
    elif camera_choice == "External USB Webcam":
        return 1
    else:
        return f"http://{mobile_ip}:4747/video"


def open_camera(source, width=320, height=240):
    """
    Open camera using OpenCV.
    Returns cv2.VideoCapture object or None if failed.
    """
    if isinstance(source, str):
        # IP camera stream (e.g., DroidCam)
        cap = cv2.VideoCapture(source)
    else:
        # Integer camera index (0,1,2...)
        if os.name == 'nt':  # Windows
            cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        else:                # Linux / macOS
            cap = cv2.VideoCapture(source)
        # Set resolution for local cameras
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if cap is None or not cap.isOpened():
        return None
    return cap


def preprocess_frame(frame, source, target_width, target_height, rotate=False):
    """
    Preprocess frame: resize for IP cameras, optionally rotate.
    """
    if isinstance(source, str):
        # IP camera: resize to target dimensions
        frame = cv2.resize(frame, (target_width, target_height))
        if rotate:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    # For local cameras, frame is already at target resolution
    return frame


def release_camera(cap):
    """Safely release the camera capture object."""
    if cap is not None:
        cap.release()


# ----------------------------------------------------------------------
# Cloud‑compatible camera functions (use st.camera_input)
# ----------------------------------------------------------------------
def get_cloud_camera_frame():
    """
    For Streamlit Cloud: returns a single frame from st.camera_input.
    Call this repeatedly in a loop to simulate live video.
    """
    img_file = st.camera_input("📸 Capture frame (click to take a picture)")
    if img_file is not None:
        # Convert uploaded file to OpenCV BGR format
        bytes_data = img_file.getvalue()
        np_arr = np.frombuffer(bytes_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return frame
    return None


# ----------------------------------------------------------------------
# Unified camera loop that works everywhere
# ----------------------------------------------------------------------
def run_camera_loop(process_frame_func, use_cloud=None, frame_width=640, frame_height=480):
    """
    Unified camera loop that works locally (OpenCV real‑time) or on cloud (st.camera_input).
    
    Parameters:
        process_frame_func: function that takes a BGR frame and returns processed frame (or None)
        use_cloud: if None, auto‑detect; if True, force cloud mode; if False, force local.
        frame_width, frame_height: target dimensions (used only in local mode)
    """
    if use_cloud is None:
        use_cloud = is_cloud()

    if use_cloud:
        st.info("🌐 Running in Cloud mode – using your browser camera. Click the button below to capture a frame. The loop will continue until you click 'Stop'.")
        # Placeholders for dynamic content
        frame_placeholder = st.empty()
        stop_button = st.button("🛑 Stop")
        
        while not stop_button:
            frame = get_cloud_camera_frame()
            if frame is not None:
                processed = process_frame_func(frame)
                if processed is not None:
                    frame_placeholder.image(processed, channels="BGR", use_container_width=True)
            time.sleep(0.1)   # small delay to avoid excessive refreshes
            # Check stop button again (needs to be re‑evaluated each iteration)
            stop_button = st.button("🛑 Stop", key=f"stop_{time.time()}")
            if stop_button:
                break
        st.success("Camera stopped.")
    else:
        # Local OpenCV mode – your original real‑time loop
        st.info("💻 Running in Local mode – using OpenCV real‑time video.")
        source = get_camera_source()
        cap = open_camera(source, width=frame_width, height=frame_height)
        if cap is None:
            st.error("❌ Could not open camera. Please check your camera connection and try again.")
            return
        
        frame_placeholder = st.empty()
        stop_button = st.button("🛑 Stop")
        
        while not stop_button:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to grab frame.")
                break
            # Preprocess (resize for IP cameras, etc.)
            frame = preprocess_frame(frame, source, frame_width, frame_height, rotate=False)
            processed = process_frame_func(frame)
            if processed is not None:
                frame_placeholder.image(processed, channels="BGR", use_container_width=True)
            # Allow UI to update and check stop button
            time.sleep(0.03)  # ~30 fps max
            stop_button = st.button("🛑 Stop", key=f"stop_local_{time.time()}")
        
        release_camera(cap)
        st.success("Camera stopped.")


# ----------------------------------------------------------------------
# Example usage (commented out – uncomment to test)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Dummy processing: just return the frame unchanged
    def test_process(frame):
        # Add a simple text overlay to show it's working
        cv2.putText(frame, "Camera OK", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return frame

    run_camera_loop(test_process)