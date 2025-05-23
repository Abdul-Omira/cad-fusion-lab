"""
Deployment Server for Text-to-CAD Model

FastAPI server for serving the Text-to-CAD model.
Supports:
- Text-to-CAD conversion
- Multiple export formats (STEP, GLTF)
- Validation and metrics
"""

import os
import sys
import logging
import argparse
import yaml
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from pathlib import Path
import time
import tempfile
import shutil
import asyncio

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.models.text_to_cad import TextToCADModel
from src.inference.pipeline import InferencePipeline, load_model_from_checkpoint
from src.validation.geometric import GeometricValidator


# Define API request/response models
class GenerateRequest(BaseModel):
    text: str
    format: str = "step"  # Options: "step", "gltf", "kcl"
    validate: bool = True
    compute_metrics: bool = False
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.95


class GenerateResponse(BaseModel):
    job_id: str
    status: str
    file_url: Optional[str] = None
    metrics: Optional[Dict[str, float]] = None
    errors: Optional[List[str]] = None


class BatchGenerateRequest(BaseModel):
    texts: List[str]
    format: str = "step"
    validate: bool = True


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: float
    file_url: Optional[str] = None
    metrics: Optional[Dict[str, float]] = None
    errors: Optional[List[str]] = None


# Setup application
app = FastAPI(
    title="Text-to-CAD API",
    description="Convert natural language descriptions to parametric CAD files",
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

# Global variables
pipeline = None
job_status = {}
output_dir = None


@app.on_event("startup")
async def startup_event():
    """Initialize model and pipeline on server startup."""
    global pipeline, output_dir
    
    # Load configuration
    config_path = os.environ.get("CONFIG_PATH", "configs/base_config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model_path = os.environ.get("MODEL_PATH", "checkpoints/final_model.pt")
    try:
        model = load_model_from_checkpoint(model_path, device)
        logging.info(f"Loaded model from {model_path}")
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        # Create a dummy model for development
        logging.warning("Creating a dummy model for development")
        model = TextToCADModel(vocab_size=10000)
    
    # Initialize pipeline
    pipeline = InferencePipeline(model, device=device, config=config.get("inference", {}))
    
    # Set up output directory
    output_dir = os.environ.get("OUTPUT_DIR", "outputs/api")
    os.makedirs(output_dir, exist_ok=True)
    
    logging.info("Server startup complete")


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Text-to-CAD API is running"}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": pipeline is not None,
        "device": pipeline.device if pipeline else "unknown"
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate_cad(request: GenerateRequest, background_tasks: BackgroundTasks):
    """Generate CAD from text description."""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Create job ID
    job_id = f"job_{int(time.time() * 1000)}"
    job_status[job_id] = {
        "status": "pending",
        "progress": 0.0,
        "file_url": None,
        "metrics": None,
        "errors": None
    }
    
    # Start generation task in background
    background_tasks.add_task(
        generate_cad_task,
        job_id,
        request.text,
        request.format,
        request.validate,
        request.compute_metrics,
        {
            "temperature": request.temperature,
            "top_k": request.top_k,
            "top_p": request.top_p
        }
    )
    
    return GenerateResponse(
        job_id=job_id,
        status="pending"
    )


async def generate_cad_task(
    job_id: str,
    text: str,
    format_type: str,
    validate: bool,
    compute_metrics: bool,
    generation_params: Dict[str, Any]
):
    """Background task for CAD generation."""
    try:
        job_status[job_id]["status"] = "processing"
        job_status[job_id]["progress"] = 0.1
        
        # Override generation parameters
        original_config = pipeline.config.copy()
        pipeline.config.update(generation_params)
        
        # Generate CAD sequence
        cad_sequence = pipeline.generate(text)
        job_status[job_id]["progress"] = 0.6
        
        # Export in requested format
        output_path = os.path.join(output_dir, f"{job_id}.{format_type}")
        
        if format_type == "step":
            file_path = pipeline.export_step(cad_sequence, output_path)
        elif format_type == "gltf":
            file_path = pipeline.export_gltf(cad_sequence, output_path)
        elif format_type == "kcl":
            kcl_code = pipeline.export_kcl(cad_sequence)
            with open(output_path, "w") as f:
                f.write(kcl_code)
            file_path = output_path
        else:
            raise ValueError(f"Unsupported format: {format_type}")
        
        job_status[job_id]["progress"] = 0.8
        
        # Compute metrics if requested
        metrics = None
        if compute_metrics:
            metrics = {
                "clip_score": pipeline.compute_visual_score(cad_sequence, text)
            }
            
            # Validate if requested
            if validate:
                validator = GeometricValidator()
                is_valid, errors = validator.validate(cad_sequence)
                metrics["is_valid"] = is_valid
                metrics["error_count"] = len(errors)
        
        job_status[job_id]["progress"] = 1.0
        job_status[job_id]["status"] = "completed"
        job_status[job_id]["file_url"] = f"/files/{job_id}.{format_type}"
        job_status[job_id]["metrics"] = metrics
        
        # Restore original config
        pipeline.config = original_config
        
    except Exception as e:
        logging.error(f"Error generating CAD for job {job_id}: {e}")
        job_status[job_id]["status"] = "failed"
        job_status[job_id]["errors"] = [str(e)]


@app.get("/status/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get status of a generation job."""
    if job_id not in job_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return JobStatusResponse(
        job_id=job_id,
        **job_status[job_id]
    )


@app.get("/files/{filename}")
async def get_file(filename: str):
    """Serve generated files."""
    file_path = os.path.join(output_dir, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    # Determine media type based on extension
    media_type = None
    if filename.endswith(".step"):
        media_type = "application/step"
    elif filename.endswith(".gltf"):
        media_type = "model/gltf+json"
    elif filename.endswith(".kcl"):
        media_type = "text/plain"
    
    return FileResponse(file_path, media_type=media_type, filename=filename)


@app.post("/batch", response_model=List[GenerateResponse])
async def batch_generate(request: BatchGenerateRequest):
    """Batch generate CAD from multiple text descriptions."""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    responses = []
    
    for text in request.texts:
        # Create job for each text
        job_id = f"job_{int(time.time() * 1000)}_{len(responses)}"
        job_status[job_id] = {
            "status": "pending",
            "progress": 0.0,
            "file_url": None,
            "metrics": None,
            "errors": None
        }
        
        responses.append(GenerateResponse(
            job_id=job_id,
            status="pending"
        ))
        
        # Start generation in background
        asyncio.create_task(generate_cad_task(
            job_id,
            text,
            request.format,
            request.validate,
            False,  # Don't compute metrics for batch jobs
            {}  # Use default generation parameters
        ))
    
    return responses


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Text-to-CAD API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host to run server on")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port to run server on")
    parser.add_argument("--model", type=str, default="checkpoints/final_model.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml",
                        help="Path to config file")
    parser.add_argument("--output-dir", type=str, default="outputs/api",
                        help="Directory to save generated files")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run model on")
    parser.add_argument("--log-level", type=str, default="info",
                        choices=["debug", "info", "warning", "error", "critical"],
                        help="Logging level")
    return parser.parse_args()


def main():
    """Main function to run the server."""
    args = parse_args()
    
    # Set environment variables
    os.environ["MODEL_PATH"] = args.model
    os.environ["CONFIG_PATH"] = args.config
    os.environ["OUTPUT_DIR"] = args.output_dir
    os.environ["DEVICE"] = args.device
    
    # Set up logging
    numeric_level = getattr(logging, args.log_level.upper(), None)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    
    # Run server
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()