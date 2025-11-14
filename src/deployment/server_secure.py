"""
Secure Deployment Server for Text-to-CAD Model

Enhanced with:
- JWT Authentication
- Rate Limiting
- Input Validation & Sanitization
- Proper CORS Configuration
- Request/Response Size Limits
- Structured Logging
- Security Headers
"""

import os
import sys
import logging
import argparse
import yaml
import torch
import uvicorn
import re
from datetime import timedelta
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Request, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
from pathlib import Path
import time
import tempfile
import shutil
import asyncio
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.models.text_to_cad import TextToCADModel
from src.inference.pipeline import InferencePipeline, load_model_from_checkpoint
from src.validation.geometric import GeometricValidator
from src.deployment.auth import (
    create_access_token,
    authenticate_user,
    get_current_active_user,
    require_scope,
    Token,
    User,
)

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(extra)s'
)
logger = logging.getLogger(__name__)

# Rate limiter
limiter = Limiter(key_func=get_remote_address)


# Input validation and sanitization
def sanitize_text(text: str) -> str:
    """Sanitize text input to prevent injection attacks."""
    # Remove any potential script tags or HTML
    text = re.sub(r'<[^>]+>', '', text)
    # Limit to printable ASCII and common unicode
    text = ''.join(char for char in text if char.isprintable() or char.isspace())
    # Trim whitespace
    text = text.strip()
    return text


# Define API request/response models with validation
class GenerateRequest(BaseModel):
    text: str = Field(..., min_length=5, max_length=500, description="Text description of CAD model")
    format: str = Field(default="step", regex="^(step|gltf|kcl)$")
    validate: bool = True
    compute_metrics: bool = False
    temperature: float = Field(default=0.8, ge=0.1, le=2.0)
    top_k: int = Field(default=50, ge=1, le=100)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)

    @validator('text')
    def sanitize_text_input(cls, v):
        """Sanitize text input."""
        return sanitize_text(v)

    class Config:
        schema_extra = {
            "example": {
                "text": "Create a rectangular bracket with mounting holes",
                "format": "step",
                "validate": True,
                "temperature": 0.8
            }
        }


class GenerateResponse(BaseModel):
    job_id: str
    status: str
    file_url: Optional[str] = None
    metrics: Optional[Dict[str, float]] = None
    errors: Optional[List[str]] = None


class BatchGenerateRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=10)
    format: str = Field(default="step", regex="^(step|gltf|kcl)$")
    validate: bool = True

    @validator('texts')
    def sanitize_texts(cls, v):
        """Sanitize all text inputs."""
        return [sanitize_text(text) for text in v]


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: float
    file_url: Optional[str] = None
    metrics: Optional[Dict[str, float]] = None
    errors: Optional[List[str]] = None


class LoginRequest(BaseModel):
    username: str
    password: str


# Setup application
app = FastAPI(
    title="Text-to-CAD API (Secure)",
    description="Convert natural language descriptions to parametric CAD files with enterprise security",
    version="1.1.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# Add rate limit exceeded handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Security: Trusted Host Middleware (prevent host header attacks)
allowed_hosts = os.getenv("ALLOWED_HOSTS", "localhost,127.0.0.1").split(",")
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=allowed_hosts
)

# Add CORS middleware with restricted origins
allowed_origins = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://localhost:8000"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,  # Specific origins only
    allow_credentials=True,
    allow_methods=["GET", "POST"],  # Specific methods only
    allow_headers=["Authorization", "Content-Type"],  # Specific headers only
    max_age=3600,  # Cache preflight requests for 1 hour
)


# Add security headers middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Add security headers to all responses."""
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Content-Security-Policy"] = "default-src 'self'"
    return response


# Add request size limit middleware
@app.middleware("http")
async def limit_request_size(request: Request, call_next):
    """Limit request body size to prevent DoS attacks."""
    max_size = 1024 * 1024  # 1MB
    if request.headers.get("content-length"):
        content_length = int(request.headers["content-length"])
        if content_length > max_size:
            return JSONResponse(
                status_code=413,
                content={"detail": "Request too large"}
            )
    return await call_next(request)


# Global variables
pipeline = None
job_status = {}
output_dir = None


@app.on_event("startup")
async def startup_event():
    """Initialize model and pipeline on server startup."""
    global pipeline, output_dir

    logger.info("Starting Text-to-CAD API server...")

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
        logger.info(f"Loaded model from {model_path}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.warning("Creating a dummy model for development")
        model = TextToCADModel(vocab_size=10000, offline_mode=True)

    # Initialize pipeline
    pipeline = InferencePipeline(model, device=device, config=config.get("inference", {}))

    # Set up output directory
    output_dir = os.environ.get("OUTPUT_DIR", "outputs/api")
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Server startup complete")


# Public endpoints (no authentication required)
@app.get("/")
@limiter.limit("10/minute")
async def root(request: Request):
    """Root endpoint."""
    return {"message": "Text-to-CAD API is running", "version": "1.1.0"}


@app.get("/health")
@limiter.limit("30/minute")
async def health(request: Request):
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": pipeline is not None,
        "device": pipeline.device if pipeline else "unknown"
    }


@app.post("/auth/login", response_model=Token)
@limiter.limit("5/minute")
async def login(request: Request, login_data: LoginRequest):
    """
    Authenticate and get access token.

    Example:
        curl -X POST "http://localhost:8000/auth/login" \\
             -H "Content-Type: application/json" \\
             -d '{"username": "admin", "password": "changeme"}'
    """
    user = authenticate_user(login_data.username, login_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=60)
    access_token = create_access_token(
        data={"sub": user.username, "scopes": user.scopes},
        expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


# Protected endpoints (require authentication)
@app.post("/generate", response_model=GenerateResponse)
@limiter.limit("10/minute")
async def generate_cad(
    request: Request,
    generate_request: GenerateRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_scope("write"))
):
    """
    Generate CAD from text description (requires authentication).

    Example:
        curl -X POST "http://localhost:8000/generate" \\
             -H "Authorization: Bearer YOUR_TOKEN" \\
             -H "Content-Type: application/json" \\
             -d '{"text": "Create a rectangular bracket", "format": "step"}'
    """
    if not pipeline:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Create job ID
    job_id = f"job_{int(time.time() * 1000)}_{current_user.username}"
    job_status[job_id] = {
        "status": "pending",
        "progress": 0.0,
        "file_url": None,
        "metrics": None,
        "errors": None,
        "user": current_user.username
    }

    # Start generation task in background
    background_tasks.add_task(
        generate_cad_task,
        job_id,
        generate_request.text,
        generate_request.format,
        generate_request.validate,
        generate_request.compute_metrics,
        {
            "temperature": generate_request.temperature,
            "top_k": generate_request.top_k,
            "top_p": generate_request.top_p
        }
    )

    logger.info(f"CAD generation started for user {current_user.username}, job {job_id}")

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

        logger.info(f"CAD generation completed for job {job_id}")

    except Exception as e:
        logger.error(f"Error generating CAD for job {job_id}: {e}")
        job_status[job_id]["status"] = "failed"
        job_status[job_id]["errors"] = [str(e)]


@app.get("/status/{job_id}", response_model=JobStatusResponse)
@limiter.limit("30/minute")
async def get_job_status(
    request: Request,
    job_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Get status of a generation job (requires authentication)."""
    if job_id not in job_status:
        raise HTTPException(status_code=404, detail="Job not found")

    # Check if user owns the job
    if job_status[job_id].get("user") != current_user.username and "admin" not in current_user.scopes:
        raise HTTPException(status_code=403, detail="Not authorized to view this job")

    return JobStatusResponse(
        job_id=job_id,
        **{k: v for k, v in job_status[job_id].items() if k != "user"}
    )


@app.get("/files/{filename}")
@limiter.limit("30/minute")
async def get_file(
    request: Request,
    filename: str,
    current_user: User = Depends(get_current_active_user)
):
    """Serve generated files (requires authentication)."""
    # Validate filename to prevent directory traversal
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    file_path = os.path.join(output_dir, filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    # Check if user owns the file (extract job_id from filename)
    job_id = filename.rsplit(".", 1)[0]
    if job_id in job_status:
        if job_status[job_id].get("user") != current_user.username and "admin" not in current_user.scopes:
            raise HTTPException(status_code=403, detail="Not authorized to access this file")

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
@limiter.limit("5/minute")
async def batch_generate(
    request: Request,
    batch_request: BatchGenerateRequest,
    current_user: User = Depends(require_scope("write"))
):
    """
    Batch generate CAD from multiple text descriptions (requires authentication).
    Limited to 10 items per batch.
    """
    if not pipeline:
        raise HTTPException(status_code=503, detail="Model not loaded")

    responses = []

    for idx, text in enumerate(batch_request.texts):
        # Create job for each text
        job_id = f"job_{int(time.time() * 1000)}_{current_user.username}_{idx}"
        job_status[job_id] = {
            "status": "pending",
            "progress": 0.0,
            "file_url": None,
            "metrics": None,
            "errors": None,
            "user": current_user.username
        }

        responses.append(GenerateResponse(
            job_id=job_id,
            status="pending"
        ))

        # Start generation in background
        asyncio.create_task(generate_cad_task(
            job_id,
            text,
            batch_request.format,
            batch_request.validate,
            False,  # Don't compute metrics for batch jobs
            {}  # Use default generation parameters
        ))

    logger.info(f"Batch generation started for user {current_user.username}, {len(batch_request.texts)} jobs")

    return responses


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Text-to-CAD Secure API Server")
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
    parser.add_argument("--reload", action="store_true",
                        help="Enable auto-reload for development")
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

    # Warning about production deployment
    if os.getenv("JWT_SECRET_KEY") == "your-secret-key-change-in-production":
        logger.warning("=" * 80)
        logger.warning("WARNING: Using default JWT secret key!")
        logger.warning("Set JWT_SECRET_KEY environment variable in production!")
        logger.warning("=" * 80)

    # Run server
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level
    )


if __name__ == "__main__":
    main()
