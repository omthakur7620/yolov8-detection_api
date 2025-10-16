"""
YOLOv8 Object Detection API with FastAPI
Advanced implementation with production-ready features
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import time
import logging
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
import io
from PIL import Image
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="YOLOv8 Object Detection API",
    description="Advanced object detection API using YOLOv8 with real-time inference",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create necessary directories
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Mount static files
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

# Global model variable for caching
MODEL_CACHE = {}

# Response models
class BoundingBox(BaseModel):
    x1: float = Field(..., description="Top-left x coordinate")
    y1: float = Field(..., description="Top-left y coordinate")
    x2: float = Field(..., description="Bottom-right x coordinate")
    y2: float = Field(..., description="Bottom-right y coordinate")
    width: float = Field(..., description="Box width")
    height: float = Field(..., description="Box height")

class Detection(BaseModel):
    class_name: str = Field(..., description="Detected object class")
    confidence: float = Field(..., description="Detection confidence score (0-1)")
    bbox: BoundingBox = Field(..., description="Bounding box coordinates")
    class_id: int = Field(..., description="Class ID from model")

class DetectionResponse(BaseModel):
    success: bool = Field(True, description="Request success status")
    detections: List[Detection] = Field(..., description="List of detected objects")
    total_detections: int = Field(..., description="Total number of detections")
    image_size: Dict[str, int] = Field(..., description="Original image dimensions")
    processing_time: float = Field(..., description="Processing time in seconds")
    model_name: str = Field(..., description="YOLO model used")
    timestamp: str = Field(..., description="Detection timestamp")
    confidence_threshold: float = Field(..., description="Confidence threshold used")
    output_image_path: Optional[str] = Field(None, description="Path to annotated image")

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    timestamp: str

def get_model(model_name: str = "yolov8n.pt") -> YOLO:
    """
    Load and cache YOLO model for efficient inference
    """
    if model_name not in MODEL_CACHE:
        logger.info(f"Loading model: {model_name}")
        try:
            MODEL_CACHE[model_name] = YOLO(model_name)
            logger.info(f"Model {model_name} loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")
    return MODEL_CACHE[model_name]

def draw_detections(image: np.ndarray, results, conf_threshold: float = 0.25) -> np.ndarray:
    """
    Draw bounding boxes and labels on image with professional styling
    """
    annotated = image.copy()
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            if box.conf[0] < conf_threshold:
                continue
                
            # Extract box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            class_name = result.names[cls]
            
            # Dynamic color based on confidence
            if conf > 0.85:
                color = (0, 255, 0)  # Green for high confidence
            elif conf > 0.7:
                color = (0, 255, 255)  # Yellow for medium confidence
            else:
                color = (0, 165, 255)  # Orange for lower confidence
            
            # Draw bounding box with thickness based on confidence
            thickness = max(2, int(conf * 4))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
            
            # Create label with background
            label = f"{class_name}: {conf:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 2
            
            # Get text size for background
            (text_width, text_height), baseline = cv2.getTextSize(
                label, font, font_scale, font_thickness
            )
            
            # Draw background rectangle for text
            cv2.rectangle(
                annotated,
                (x1, y1 - text_height - 10),
                (x1 + text_width + 10, y1),
                color,
                -1
            )
            
            # Draw text
            cv2.putText(
                annotated,
                label,
                (x1 + 5, y1 - 5),
                font,
                font_scale,
                (0, 0, 0),
                font_thickness,
                cv2.LINE_AA
            )
            
            # Add small circle at center of box
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            cv2.circle(annotated, (center_x, center_y), 4, color, -1)
    
    return annotated

@app.on_event("startup")
async def startup_event():
    """
    Load model on startup for faster first request
    """
    logger.info("Starting up API...")
    try:
        get_model("yolov8n.pt")
        logger.info("Startup complete - Model preloaded")
    except Exception as e:
        logger.error(f"Startup error: {e}")

@app.get("/", response_model=HealthResponse)
async def root():
    """
    Root endpoint - Health check
    """
    model_loaded = "yolov8n.pt" in MODEL_CACHE
    return HealthResponse(
        status="online",
        model_loaded=model_loaded,
        model_name="yolov8n.pt" if model_loaded else "not_loaded",
        timestamp=datetime.now().isoformat()
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Detailed health check endpoint
    """
    model_loaded = "yolov8n.pt" in MODEL_CACHE
    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        model_name="yolov8n.pt" if model_loaded else "not_loaded",
        timestamp=datetime.now().isoformat()
    )

@app.post("/detect", response_model=DetectionResponse)
async def detect_objects(
    file: UploadFile = File(..., description="Image file for object detection"),
    conf_threshold: float = 0.25,
    save_output: bool = True,
    model_name: str = "yolov8n.pt"
):
    """
    Main detection endpoint - Accepts image and returns detections
    
    Parameters:
    - file: Image file (JPG, PNG, etc.)
    - conf_threshold: Minimum confidence threshold (0-1)
    - save_output: Whether to save annotated output image
    - model_name: YOLO model to use (default: yolov8n.pt)
    
    Returns:
    - JSON with detections, confidence scores, bounding boxes, and metadata
    """
    start_time = time.time()
    
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Please upload an image."
        )
    
    try:
        # Read and decode image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Failed to decode image")
        
        original_height, original_width = image.shape[:2]
        logger.info(f"Processing image: {file.filename} ({original_width}x{original_height})")
        
        # Get model and run inference
        model = get_model(model_name)
        results = model(image, conf=conf_threshold, verbose=False)
        
        # Parse results
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = result.names[cls]
                
                detection = Detection(
                    class_name=class_name,
                    confidence=round(conf, 4),
                    bbox=BoundingBox(
                        x1=round(x1, 2),
                        y1=round(y1, 2),
                        x2=round(x2, 2),
                        y2=round(y2, 2),
                        width=round(x2 - x1, 2),
                        height=round(y2 - y1, 2)
                    ),
                    class_id=cls
                )
                detections.append(detection)
        
        # Save annotated image
        output_path = None
        if save_output and detections:
            annotated_image = draw_detections(image, results, conf_threshold)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"detection_{timestamp}_{file.filename}"
            output_path = OUTPUT_DIR / output_filename
            cv2.imwrite(str(output_path), annotated_image)
            output_path = f"/outputs/{output_filename}"
            logger.info(f"Saved output to: {output_path}")
        
        processing_time = time.time() - start_time
        
        response = DetectionResponse(
            success=True,
            detections=detections,
            total_detections=len(detections),
            image_size={"width": original_width, "height": original_height},
            processing_time=round(processing_time, 4),
            model_name=model_name,
            timestamp=datetime.now().isoformat(),
            confidence_threshold=conf_threshold,
            output_image_path=output_path
        )
        
        logger.info(f"Detection complete: {len(detections)} objects found in {processing_time:.3f}s")
        return response
        
    except Exception as e:
        logger.error(f"Detection error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@app.post("/detect-batch")
async def detect_batch(
    files: List[UploadFile] = File(..., description="Multiple image files"),
    conf_threshold: float = 0.25
):
    """
    Batch detection endpoint - Process multiple images at once
    """
    results = []
    
    for file in files:
        try:
            # Reuse the detect_objects logic
            result = await detect_objects(file, conf_threshold, save_output=False)
            results.append({
                "filename": file.filename,
                "success": True,
                "result": result
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })
    
    return {
        "total_images": len(files),
        "successful": sum(1 for r in results if r["success"]),
        "failed": sum(1 for r in results if not r["success"]),
        "results": results
    }

@app.get("/model-info")
async def model_info():
    """
    Get information about the loaded model
    """
    if "yolov8n.pt" not in MODEL_CACHE:
        return {"error": "Model not loaded"}
    
    model = MODEL_CACHE["yolov8n.pt"]
    return {
        "model_name": "yolov8n.pt",
        "classes": model.names,
        "num_classes": len(model.names),
        "model_loaded": True
    }

if __name__ == "__main__":
    logger.info("Starting YOLOv8 Detection API Server...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )