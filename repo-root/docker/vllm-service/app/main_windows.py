"""
Windows-Compatible Qwen Inference Service (Transformers-based)

This service runs on Windows with AMD ROCm GPUs and provides:
- Text-based LLM inference via REST API
- Vision OCR via REST API
- Compatible with the existing vLLM client

Environment Variables:
- VLLM_TEXT_MODEL: Text model name (default: Qwen/Qwen2.5-7B-Instruct)
- VLLM_VISION_MODEL: Vision model name (default: Qwen/Qwen2-VL-7B-Instruct)
- VLLM_GPU_MEMORY_UTIL: GPU memory utilization (default: 0.85)
- ENABLE_VISION_OCR: Enable vision model (default: true)
- PORT: Service port (default: 8000)
- HF_TOKEN: Hugging Face token for model downloads
"""

import os
import logging
import base64
import json
import asyncio
from typing import List, Optional
from contextlib import asynccontextmanager, nullcontext
from io import BytesIO

# Ensure PyTorch uses expandable segments to reduce OOM fragmentation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoProcessor,
    AutoModelForVision2Seq,
)
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global model instances
text_model = None
text_tokenizer = None
vision_model = None
vision_processor = None
device = None


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
    global text_model, text_tokenizer, vision_model, vision_processor, device

    text_model_name = os.getenv("VLLM_TEXT_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    vision_model_name = os.getenv("VLLM_VISION_MODEL", "Qwen/Qwen2-VL-7B-Instruct")
    enable_vision = os.getenv("ENABLE_VISION_OCR", "true").lower() == "true"

    logger.info("Initializing Qwen Inference Service (Windows/Transformers)...")
    logger.info(f"  Text Model: {text_model_name}")
    logger.info(f"  Vision Model: {vision_model_name if enable_vision else 'Disabled'}")

    try:
        # Detect GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"  Device: {device}")

        if torch.cuda.is_available():
            logger.info(f"  GPU Count: {torch.cuda.device_count()}")
            logger.info(f"  GPU 0: {torch.cuda.get_device_name(0)}")
            if torch.cuda.device_count() > 1:
                logger.info(f"  GPU 1: {torch.cuda.get_device_name(1)}")

        # Load text model
        logger.info(f"Loading text model: {text_model_name}...")
        text_tokenizer = AutoTokenizer.from_pretrained(
            text_model_name,
            trust_remote_code=True
        )
        text_model = AutoModelForCausalLM.from_pretrained(
            text_model_name,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True
        )
        logger.info("✓ Text model ready")

        # Load vision model if enabled
        if enable_vision:
            logger.info(f"Loading vision model: {vision_model_name}...")
            vision_processor = AutoProcessor.from_pretrained(
                vision_model_name,
                trust_remote_code=True
            )
            vision_model = AutoModelForVision2Seq.from_pretrained(
                vision_model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
                trust_remote_code=True
            )
            logger.info("✓ Vision model ready")

        logger.info("Qwen Inference Service is READY")

    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        raise

    yield

    logger.info("Shutting down...")


app = FastAPI(title="Qwen Inference Service (Windows)", lifespan=lifespan)


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/health")
async def health():
    """Health check - Streamlit UI uses this to verify service is up."""
    return {
        "status": "healthy",
        "text_model_loaded": text_model is not None,
        "vision_model_loaded": vision_model is not None,
        "gpu_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count(),
        "device": str(device)
    }


@app.post("/api/v1/query")
async def query(request: TextQueryRequest):
    """
    Single text query - Used by Streamlit UI for layout analysis, etc.
    """
    if not text_model:
        raise HTTPException(503, "Model not ready")

    try:
        # Prepare messages format for Qwen2.5
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": request.prompt}
        ]

        text = text_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = text_tokenizer([text], return_tensors="pt").to(device)

        # Generate
        with torch.no_grad():
            generated_ids = text_model.generate(
                **model_inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                do_sample=request.temperature > 0,
            )

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = text_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return {"text": response}

    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(500, str(e))


@app.post("/api/v1/batch")
async def batch(request: BatchQueryRequest):
    """
    Batch queries - Used by Streamlit UI to process multiple pages at once.
    """
    if not text_model:
        raise HTTPException(503, "Model not ready")

    try:
        results = []
        for prompt in request.prompts:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]

            text = text_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            model_inputs = text_tokenizer([text], return_tensors="pt").to(device)

            with torch.no_grad():
                generated_ids = text_model.generate(
                    **model_inputs,
                    max_new_tokens=request.max_tokens,
                    temperature=request.temperature,
                    do_sample=request.temperature > 0,
                )

            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            response = text_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            results.append(response)

        return {"results": results}

    except Exception as e:
        logger.error(f"Batch failed: {e}")
        raise HTTPException(500, str(e))


@app.post("/api/v1/ocr")
async def ocr(request: OCRRequest):
    """
    Vision OCR - Used by Streamlit UI for scanned documents.
    """
    if not vision_model:
        raise HTTPException(503, "Vision model not enabled")

    try:
        # Decode image
        img_data = base64.b64decode(request.image_base64)
        image = Image.open(BytesIO(img_data)).convert("RGB")

        # OCR prompt for Qwen2-VL
        if request.focus_technical:
            prompt = """Extract all text from this engineering/technical document image.
Focus on:
- Dimensions and measurements
- Part numbers and identifiers
- Labels and annotations
- Technical specifications

Return JSON format:
{"texts": [{"text": "...", "bbox": [x0, y0, x1, y1], "confidence": 0.95}]}

Where bbox coordinates are percentages (0-100) of image width/height."""
        else:
            prompt = """Extract all visible text from this image.
Return JSON format:
{"texts": [{"text": "...", "bbox": [x0, y0, x1, y1], "confidence": 0.95}]}

Where bbox coordinates are percentages (0-100) of image width/height."""

        # Prepare inputs for Qwen2-VL
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Process
        text = vision_processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = vision_processor(
            images=image,
            text=[text],
            padding=True,
            return_tensors="pt"
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            autocast_ctx = (
                torch.autocast(device_type=device.type, dtype=torch.float16)
                if device.type == "cuda"
                else nullcontext()
            )
            with autocast_ctx:
                generated_ids = vision_model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.1,
                )

        input_length = inputs["input_ids"].shape[-1]
        generated_ids_trimmed = generated_ids[:, input_length:]

        output_text = vision_processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        # Parse JSON response
        try:
            result = json.loads(output_text)
            texts = result.get("texts", [])
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON from model output: {output_text[:200]}")
            # Fallback: return raw text
            texts = [{"text": output_text, "bbox": [0, 0, 100, 100], "confidence": 0.8}]

        # Filter by confidence
        filtered = [
            t for t in texts
            if t.get("confidence", 0) >= request.min_confidence
        ]

        if device.type == "cuda":
            torch.cuda.empty_cache()

        return {"texts": filtered, "total": len(filtered)}

    except Exception as e:
        logger.error(f"OCR failed: {e}", exc_info=True)
        raise HTTPException(500, str(e))


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    try:
        uvicorn.run(app, host="0.0.0.0", port=port)
    except KeyboardInterrupt:
        logger.info("Shutdown signal received. Exiting gracefully.")
    except asyncio.CancelledError:
        logger.info("Async tasks cancelled during shutdown. Exiting gracefully.")
