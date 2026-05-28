# import cv2
# import streamlit as st
# import os

# def get_camera_source():
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
#         return f"http://192.168.1.22:4747/video"

# def open_camera(source, width, height):
#     if isinstance(source, str):
#         cap = cv2.VideoCapture(source)
#         # For URL streams, we'll resize later
#     else:
#         if os.name == 'nt':
#             cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
#         else:
#             cap = cv2.VideoCapture(source)
#         cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
#         cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
#     return cap

# def preprocess_frame(frame, source, width, height):
#     if isinstance(source, str):
#         frame = cv2.resize(frame, (width, height))
#         # Uncomment if rotation needed: frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
#     return frame
"""
camera_utils.py – Optimized for all three apps (Gesture, Fire, VisionMate)
Supports:
- Laptop integrated camera (index 0)
- External USB webcam (index 1, with CAP_DSHOW on Windows)
- Mobile IP camera (DroidCam, IP Webcam) via URL
"""

import cv2
import streamlit as st
import os

def get_camera_source():
    """Display sidebar widgets to select camera source."""
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
        # DroidCam HTTP stream URL (default port 4747)
        return f"http://192.168.1.21:4747/video"

def open_camera(source, width=320, height=240):
    """
    Open camera source with proper backend.
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
    
    if not cap.isOpened():
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