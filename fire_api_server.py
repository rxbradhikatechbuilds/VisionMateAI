# app.py - Fire Detection API using FastAPI (FIXED VERSION)

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import torch
import numpy as np
from ultralytics import YOLO
import cv2
from PIL import Image
import io
import os
import uuid
from datetime import datetime
import shutil
import logging
from pathlib import Path
import json
import time  # Moved import to top

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Fire Detection API",
    description="API for detecting fire in images using YOLOv8",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
class Config:
    MODEL_PATH = "pages/models/best.pt"
    #MODEL_PATH = "E:\\FinalvisionMateAi-V2\\FinalvisionMateAi-V2\\pages\\models\\best.pt"
    ONNX_MODEL_PATH = "pages/models/fire_detection_best.onnx"
    #ONNX_MODEL_PATH = "E:\\FinalvisionMateAi-V2\\FinalvisionMateAi-V2\\pages\\models\\fire_detection_best.onnx"
    CONFIDENCE_THRESHOLD = 0.35  # Changed from 0.25 to 0.35 (hardcoded)
    IOU_THRESHOLD = 0.50  # Changed from 0.45 to 0.50 (hardcoded)
    MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    TEMP_DIR = "temp_uploads"
    
    # Undetermined detection thresholds
    LOW_CONFIDENCE_THRESHOLD = 0.15  # Below this is "undetermined"
    AMBIGUOUS_THRESHOLD = 0.25  # Changed from 0.35 to 0.25

config = Config()

# Create temp directory if not exists
os.makedirs(config.TEMP_DIR, exist_ok=True)

# Global model variable
model = None

# Response Models
class DetectionResult(BaseModel):
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]
    class_name: str
    class_id: int

class PredictionResponse(BaseModel):
    success: bool
    message: str
    status: str  # "fire_detected", "no_fire", "undetermined", "uncertain", "error"
    detections: List[DetectionResult]
    num_detections: int
    max_confidence: float
    image_id: str
    processing_time_ms: float
    timestamp: str
    additional_info: Optional[Dict[str, Any]] = None

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str
    device: str
    timestamp: str

# Load model on startup
@app.on_event("startup")
async def load_model():
    global model
    try:
        # Try to load ONNX model first (faster)
        if os.path.exists(config.ONNX_MODEL_PATH):
            logger.info(f"Loading ONNX model from {config.ONNX_MODEL_PATH}")
            model = YOLO(config.ONNX_MODEL_PATH)
        elif os.path.exists(config.MODEL_PATH):
            logger.info(f"Loading PyTorch model from {config.MODEL_PATH}")
            model = YOLO(config.MODEL_PATH)
        else:
            logger.error(f"Model not found at {config.MODEL_PATH}")
            model = None
            return
        
        # Determine device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Model loaded successfully on {device}")
        logger.info(f"Using thresholds - Confidence: {config.CONFIDENCE_THRESHOLD}, IoU: {config.IOU_THRESHOLD}")
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        model = None

@app.on_event("shutdown")
async def cleanup():
    """Clean up temporary files"""
    try:
        if os.path.exists(config.TEMP_DIR):
            shutil.rmtree(config.TEMP_DIR)
            logger.info("Cleaned up temporary files")
    except Exception as e:
        logger.error(f"Cleanup error: {str(e)}")

# Helper functions
def validate_image_file(file: UploadFile) -> bool:
    """Validate uploaded image file"""
    # Check file extension
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in config.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file format. Allowed formats: {', '.join(config.ALLOWED_EXTENSIONS)}"
        )
    return True

def determine_prediction_status(confidence: float, num_detections: int) -> str:
    """
    Determine the status of prediction based on confidence scores
    Returns: "fire_detected", "undetermined", "uncertain", "no_fire"
    """
    if num_detections == 0:
        return "no_fire"
    
    max_conf = confidence
    
    if max_conf >= config.CONFIDENCE_THRESHOLD:
        return "fire_detected"
    elif max_conf >= config.AMBIGUOUS_THRESHOLD:
        return "uncertain"
    elif max_conf > 0:
        return "undetermined"
    else:
        return "no_fire"

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint for API health check"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return HealthResponse(
        status="online",
        model_loaded=model is not None,
        model_path=config.MODEL_PATH if os.path.exists(config.MODEL_PATH) else "Not found",
        device=device,
        timestamp=datetime.now().isoformat()
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return HealthResponse(
        status="online" if model is not None else "degraded",
        model_loaded=model is not None,
        model_path=config.MODEL_PATH if os.path.exists(config.MODEL_PATH) else "Not found",
        device=device,
        timestamp=datetime.now().isoformat()
    )

# FIXED: Removed confidence_threshold and iou_threshold parameters - now using hardcoded values
@app.post("/predict", response_model=PredictionResponse)
async def predict_fire(
    file: UploadFile = File(..., description="Image file to analyze")
    # REMOVED: confidence_threshold and iou_threshold parameters
):
    """
    Predict fire in uploaded image
    
    Uses FIXED thresholds:
    - Confidence threshold: 35%
    - IoU threshold: 50%
    
    Returns:
        - fire_detected: Fire confidently detected (≥35% confidence)
        - uncertain: Possible fire but low confidence (25-35%)
        - undetermined: Very weak signal (15-25%)
        - no_fire: No fire detected
    """
    
    # Validate model is loaded
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please try again later.")
    
    # Validate image
    validate_image_file(file)
    
    # Generate unique ID for this prediction
    image_id = str(uuid.uuid4())
    temp_path = None
    
    try:
        # Read and validate file size
        contents = await file.read()
        if len(contents) > config.MAX_IMAGE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: {config.MAX_IMAGE_SIZE / 1024 / 1024:.0f}MB"
            )
        
        # Save temporary file
        file_ext = os.path.splitext(file.filename)[1]
        temp_path = os.path.join(config.TEMP_DIR, f"{image_id}{file_ext}")
        
        with open(temp_path, "wb") as f:
            f.write(contents)
        
        # Validate image can be opened
        try:
            test_image = Image.open(io.BytesIO(contents))
            test_image.verify()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
        
        # Perform inference with HARDCODED thresholds
        start_time = time.time()
        
        # FIXED: Using config values directly (hardcoded)
        results = model(
            temp_path,
            conf=config.CONFIDENCE_THRESHOLD,  # Hardcoded to 0.35
            iou=config.IOU_THRESHOLD,          # Hardcoded to 0.50
            verbose=False
        )
        
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        # Process results
        detections = []
        max_confidence = 0.0
        
        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:  # FIXED: Added length check
                boxes = result.boxes
                for box in boxes:
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = result.names[class_id]
                    bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                    
                    detections.append(DetectionResult(
                        confidence=confidence,
                        bbox=bbox,
                        class_name=class_name,
                        class_id=class_id
                    ))
                    
                    if confidence > max_confidence:
                        max_confidence = confidence
        
        # Determine prediction status
        status = determine_prediction_status(max_confidence, len(detections))
        
        # Create appropriate message
        if status == "fire_detected":
            message = f"🔥 Fire detected with {max_confidence*100:.1f}% confidence"
        elif status == "uncertain":
            message = f"⚠️ Possible fire detected but confidence is low ({max_confidence*100:.1f}%). Manual review recommended."
        elif status == "undetermined":
            message = f"❓ Undetermined: Very weak fire signal detected ({max_confidence*100:.1f}%). Could be false positive. Manual inspection required."
        elif status == "no_fire":
            message = "✅ No fire detected in the image"
        else:
            message = "Prediction completed"
        
        # Additional info for undetermined cases
        additional_info = {}
        if status in ["undetermined", "uncertain"]:
            additional_info = {
                "review_required": True,
                "suggested_action": "Manual verification needed",
                "confidence_level": "low" if status == "undetermined" else "medium-low",
                "thresholds": {
                    "detection_threshold": config.CONFIDENCE_THRESHOLD,
                    "ambiguous_threshold": config.AMBIGUOUS_THRESHOLD,
                    "current_confidence": max_confidence
                }
            }
        
        return PredictionResponse(
            success=True,
            message=message,
            status=status,
            detections=detections,
            num_detections=len(detections),
            max_confidence=max_confidence,
            image_id=image_id,
            processing_time_ms=round(processing_time, 2),
            timestamp=datetime.now().isoformat(),
            additional_info=additional_info if additional_info else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    finally:
        # Clean up temp file
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e:
                logger.warning(f"Failed to remove temp file {temp_path}: {str(e)}")

# FIXED: Batch prediction endpoint
@app.post("/predict-batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    """
    Batch prediction for multiple images
    """
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 images per batch request")
    
    results = []
    for file in files:
        try:
            # FIXED: Call predict_fire correctly without threshold parameters
            result = await predict_fire(file)
            results.append({
                "filename": file.filename,
                "result": result.dict()
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })
    
    return {
        "success": True,
        "total_images": len(files),
        "processed": len(results),
        "results": results,
        "timestamp": datetime.now().isoformat()
    }

# FIXED: Predict from path endpoint
@app.post("/predict-from-path")
async def predict_from_path(image_path: str):
    """
    Predict fire from local image path (useful for testing)
    """
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail=f"Image not found at {image_path}")
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        start_time = time.time()
        # FIXED: Using hardcoded thresholds
        results = model(image_path, conf=config.CONFIDENCE_THRESHOLD, iou=config.IOU_THRESHOLD)
        processing_time = (time.time() - start_time) * 1000
        
        detections = []
        max_confidence = 0.0
        
        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:  # FIXED: Added length check
                boxes = result.boxes
                for box in boxes:
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = result.names[class_id]
                    bbox = box.xyxy[0].tolist()
                    
                    detections.append(DetectionResult(
                        confidence=confidence,
                        bbox=bbox,
                        class_name=class_name,
                        class_id=class_id
                    ))
                    
                    if confidence > max_confidence:
                        max_confidence = confidence
        
        status = determine_prediction_status(max_confidence, len(detections))
        
        return PredictionResponse(
            success=True,
            message=f"Prediction completed with status: {status}",
            status=status,
            detections=detections,
            num_detections=len(detections),
            max_confidence=max_confidence,
            image_id=str(uuid.uuid4()),
            processing_time_ms=round(processing_time, 2),
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# NEW: Endpoint to get current threshold settings
@app.get("/thresholds")
async def get_thresholds():
    """Get current threshold settings (read-only)"""
    return {
        "confidence_threshold": config.CONFIDENCE_THRESHOLD,
        "iou_threshold": config.IOU_THRESHOLD,
        "ambiguous_threshold": config.AMBIGUOUS_THRESHOLD,
        "low_confidence_threshold": config.LOW_CONFIDENCE_THRESHOLD,
        "description": "These thresholds are fixed and cannot be changed via API"
    }

# Command line interface for local testing
if __name__ == "__main__":
    import uvicorn
    
    print("="*50)
    print("Fire Detection API Server")
    print("="*50)
    print(f"Model path: {config.MODEL_PATH}")
    print(f"Confidence Threshold: {config.CONFIDENCE_THRESHOLD} (35%)")
    print(f"IoU Threshold: {config.IOU_THRESHOLD} (50%)")
    print(f"Model loaded: {model is not None}")
    print("\nStarting server...")
    print("API Documentation: http://localhost:8000/docs")
    print("="*50)
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
