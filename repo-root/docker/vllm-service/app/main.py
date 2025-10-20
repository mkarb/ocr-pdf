"""
vLLM Microservice for PDF Analysis - AMD ROCm

This service runs SEPARATELY from the Streamlit UI and provides:
- Text-based LLM inference via REST API
- Vision OCR via REST API
- The Streamlit UI sends HTTP requests to this service

Environment Variables:
- VLLM_TEXT_MODEL: Text model name (default: Qwen/Qwen2.5-7B-Instruct)
- VLLM_VISION_MODEL: Vision model name (default: Qwen/Qwen2-VL-7B-Instruct)
- VLLM_TENSOR_PARALLEL: Number of GPUs to use (default: 2)
- VLLM_GPU_MEMORY_UTIL: GPU memory utilization (default: 0.85)
- ENABLE_VISION_OCR: Enable vision model (default: true)
- PORT: Service port (default: 8000)
"""

import os
import logging
import base64
import json
from typing import List, Optional
from contextlib import asynccontextmanager
from io import BytesIO

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global vLLM instances
vllm_text_model = None
vllm_vision_model = None


# ============================================================================
# Request/Response Models
# ============================================================================

class TextQueryRequest(BaseModel):
    prompt: str
    temperature: float = 0.1
    max_tokens: int = 512

class BatchQueryRequest(BaseModel):
    prompts: List[str]
    temperature: float = 0.1
    max_tokens: int = 512

class OCRRequest(BaseModel):
    image_base64: str
    focus_technical: bool = True
    min_confidence: float = 0.5


# ============================================================================
# Startup/Shutdown
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize models on startup."""
    global vllm_text_model, vllm_vision_model

    text_model = os.getenv("VLLM_TEXT_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    vision_model = os.getenv("VLLM_VISION_MODEL", "Qwen/Qwen2-VL-7B-Instruct")
    tensor_parallel = int(os.getenv("VLLM_TENSOR_PARALLEL", "2"))
    gpu_memory = float(os.getenv("VLLM_GPU_MEMORY_UTIL", "0.85"))
    enable_vision = os.getenv("ENABLE_VISION_OCR", "true").lower() == "true"

    logger.info("Initializing vLLM Service...")
    logger.info(f"  Text Model: {text_model}")
    logger.info(f"  Vision Model: {vision_model if enable_vision else 'Disabled'}")

    try:
        from vllm import LLM
        import torch

        logger.info(f"  GPU Available: {torch.cuda.is_available()}")
        logger.info(f"  GPU Count: {torch.cuda.device_count()}")

        # Load text model
        logger.info("Loading text model...")
        vllm_text_model = LLM(
            model=text_model,
            tensor_parallel_size=tensor_parallel,
            gpu_memory_utilization=gpu_memory,
            dtype="float16",
            trust_remote_code=True,
        )
        logger.info("✓ Text model ready")

        # Load vision model if enabled
        if enable_vision:
            logger.info("Loading vision model...")
            vllm_vision_model = LLM(
                model=vision_model,
                tensor_parallel_size=tensor_parallel,
                gpu_memory_utilization=gpu_memory,
                dtype="float16",
                trust_remote_code=True,
            )
            logger.info("✓ Vision model ready")

        logger.info("vLLM Service is READY")

    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        raise

    yield

    logger.info("Shutting down...")


app = FastAPI(title="vLLM PDF Analysis Service", lifespan=lifespan)


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/health")
async def health():
    """Health check - Streamlit UI uses this to verify service is up."""
    import torch
    return {
        "status": "healthy",
        "text_model_loaded": vllm_text_model is not None,
        "vision_model_loaded": vllm_vision_model is not None,
        "gpu_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count()
    }


@app.post("/api/v1/query")
async def query(request: TextQueryRequest):
    """
    Single text query - Used by Streamlit UI for layout analysis, etc.

    Example from UI:
        response = requests.post(
            "http://vllm-service:8000/api/v1/query",
            json={"prompt": "Analyze this layout...", "temperature": 0.1}
        )
        result = response.json()["text"]
    """
    if not vllm_text_model:
        raise HTTPException(503, "Model not ready")

    try:
        from vllm import SamplingParams

        params = SamplingParams(
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )

        outputs = vllm_text_model.generate([request.prompt], params)
        return {"text": outputs[0].outputs[0].text}

    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(500, str(e))


@app.post("/api/v1/batch")
async def batch(request: BatchQueryRequest):
    """
    Batch queries - Used by Streamlit UI to process multiple pages at once.

    Example from UI:
        prompts = [f"Analyze page {i}..." for i in range(10)]
        response = requests.post(
            "http://vllm-service:8000/api/v1/batch",
            json={"prompts": prompts}
        )
        results = response.json()["results"]
    """
    if not vllm_text_model:
        raise HTTPException(503, "Model not ready")

    try:
        from vllm import SamplingParams

        params = SamplingParams(
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )

        outputs = vllm_text_model.generate(request.prompts, params)
        return {"results": [o.outputs[0].text for o in outputs]}

    except Exception as e:
        logger.error(f"Batch failed: {e}")
        raise HTTPException(500, str(e))


@app.post("/api/v1/ocr")
async def ocr(request: OCRRequest):
    """
    Vision OCR - Used by Streamlit UI for scanned documents.

    Example from UI:
        # In highres_ocr.py or Streamlit UI:
        import base64
        with open("page.png", "rb") as f:
            img_base64 = base64.b64encode(f.read()).decode()

        response = requests.post(
            "http://vllm-service:8000/api/v1/ocr",
            json={"image_base64": img_base64, "focus_technical": True}
        )
        texts = response.json()["texts"]
    """
    if not vllm_vision_model:
        raise HTTPException(503, "Vision model not enabled")

    try:
        from vllm import SamplingParams
        from PIL import Image

        # Decode image
        img_data = base64.b64decode(request.image_base64)
        image = Image.open(BytesIO(img_data)).convert("RGB")

        # OCR prompt
        prompt = """Extract all text from this image.
Return JSON: {"texts": [{"text": "...", "bbox": [x0,y0,x1,y1], "confidence": 0.95}]}
Bbox values are percentages (0-100)."""

        params = SamplingParams(temperature=0.1, max_tokens=2048)

        outputs = vllm_vision_model.generate(
            [{"prompt": prompt, "multi_modal_data": {"image": image}}],
            params
        )

        result = json.loads(outputs[0].outputs[0].text)

        # Filter by confidence
        filtered = [
            t for t in result.get("texts", [])
            if t.get("confidence", 0) >= request.min_confidence
        ]

        return {"texts": filtered, "total": len(filtered)}

    except Exception as e:
        logger.error(f"OCR failed: {e}")
        raise HTTPException(500, str(e))


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
