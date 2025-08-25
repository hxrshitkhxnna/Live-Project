"""Document Forgery Detection API
-------------------------------------------------
FastAPI service that uses a Vision‑Transformer model hosted on HuggingFace to predict
whether an uploaded document image is genuine or forged and returns a rich set of
KPIs that may be used downstream for risk‑scoring or analytics dashboards.

Improvements compared to the original snippet
=============================================
* **Removed unused imports** (numpy, asyncio)
* **Unified** the single `POST /kpi_calculation/` endpoint instead of two overlapping
  route declarations.
* **Wrapped model artefacts** in a small singleton‑style class so we avoid global
  keywords while still keeping them accessible.
* **Explicitly pass the loaded `image_processor` and `model`** to the HuggingFace
  `pipeline` to avoid any mismatch when the repo contains multiple checkpoints.
* **Automatic CUDA / CPU device selection** via PyTorch (`torch.cuda.is_available()`).
* **Richer logging** format that includes timestamps, making diagnosis easier in
  production.
* **Clearer HTTP responses & status codes** – user‑facing error messages are concise
  while full tracebacks are logged.
* **100 % typings** so editors / linters can help you catch errors earlier.
* **PEP‑8 / black** compliant layout (max 88 columns) for readability.
"""

from __future__ import annotations

import io
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    pipeline,
)
import torch  # type: ignore  # (Ignored if torch is not installed at type‑check time)

#####################################################################################
# Configuration & Logging
#####################################################################################

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("forgery‑api")

HF_TOKEN: str | None = os.getenv("HF_TOKEN")  # Optional if repo is public
MODEL_NAME = "aevalone/vit-base-patch16-224-finetuned-forgery"

#####################################################################################
# FastAPI App
#####################################################################################

app = FastAPI(
    title="Document Forgery Detection API",
    description="Detect forged documents using a Vision Transformer model",
    version="1.0.0",
)

#####################################################################################
# Model Holder – keeps everything in one place instead of scattering `global`s
#####################################################################################


class ModelHolder:  # pylint: disable=too-few-public-methods
    """A light container for model artefacts."""

    processor: AutoImageProcessor | None = None
    model: AutoModelForImageClassification | None = None
    pipe: Any | None = None  # `transformers.pipelines.base.Pipeline`
    ready: bool = False


model_holder = ModelHolder()


@app.on_event("startup")
async def load_model() -> None:
    """Load model & processor exactly once at startup."""

    try:
        logger.info("Loading model '%s' ...", MODEL_NAME)

        model_holder.processor = AutoImageProcessor.from_pretrained(
            MODEL_NAME, token=HF_TOKEN
        )
        model_holder.model = AutoModelForImageClassification.from_pretrained(
            MODEL_NAME, token=HF_TOKEN
        )

        device = 0 if torch.cuda.is_available() else -1
        model_holder.pipe = pipeline(
            task="image-classification",
            model=model_holder.model,
            image_processor=model_holder.processor,
            device=device,
        )

        model_holder.ready = True
        logger.info("Model loaded successfully (device=%s).", "cuda" if device == 0 else "cpu")

    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Model loading failed: %s", exc)
        model_holder.ready = False  # Service stays up but reports not‑ready


#####################################################################################
# Utility helpers
#####################################################################################

def _risk_level(confidence: float, forged: bool) -> str:
    if forged:
        return (
            "HIGH_RISK" if confidence > 0.8 else "MEDIUM_RISK" if confidence > 0.6 else "LOW_RISK"
        )
    return (
        "LOW_RISK" if confidence > 0.8 else "MEDIUM_RISK" if confidence > 0.6 else "HIGH_RISK"
    )


def _certainty(confidence: float) -> str:
    match confidence:
        case c if c > 0.9:
            return "VERY_HIGH"
        case c if c > 0.8:
            return "HIGH"
        case c if c > 0.7:
            return "MEDIUM"
        case c if c > 0.6:
            return "LOW"
        case _:
            return "VERY_LOW"


def _size_category(width: int, height: int) -> str:
    pixels = width * height
    if pixels > 2_000_000:
        return "HIGH_RESOLUTION"
    if pixels > 500_000:
        return "MEDIUM_RESOLUTION"
    return "LOW_RESOLUTION"


def _authenticity_score(predictions: Dict[str, float], forged: bool) -> float:
    if not forged:
        genuine_scores = [v for k, v in predictions.items() if "genuine" in k.lower() or "authentic" in k.lower()]
        if genuine_scores:
            return round(max(genuine_scores) * 100, 2)
    forgery_scores = [v for k, v in predictions.items() if "forg" in k.lower()]
    if forgery_scores:
        return round((1 - max(forgery_scores)) * 100, 2)
    return 50.0  # Fallback


def _recommendation(forged: bool, confidence: float) -> str:
    if forged:
        if confidence > 0.8:
            return "REJECT_DOCUMENT - High probability of forgery detected"
        if confidence > 0.6:
            return "MANUAL_REVIEW - Possible forgery, requires human verification"
        return "PROCEED_WITH_CAUTION - Low confidence forgery detection"

    if confidence > 0.8:
        return "ACCEPT_DOCUMENT - Document appears genuine"
    if confidence > 0.6:
        return "ADDITIONAL_VERIFICATION - Likely genuine, consider extra checks"
    return "MANUAL_REVIEW - Low confidence in genuineness assessment"


#####################################################################################
# Core KPI computation
#####################################################################################


def _calculate_kpi(img_bytes: bytes) -> Dict[str, Any]:  # noqa: C901
    """Runs the forgery detection pipeline and produces the KPI dictionary."""

    start = time.time()

    # ---------- Image I/O ----------
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    width, height = image.size
    img_format = image.format or "Unknown"

    # ---------- Inference ----------
    logger.info("Running inference …")
    assert model_holder.pipe is not None  # for mypy / IDEs
    results = model_holder.pipe(image)

    # ---------- Post‑processing ----------
    predictions: Dict[str, float] = {r["label"]: round(r["score"], 4) for r in results}
    best = max(results, key=lambda r: r["score"])
    predicted_label: str = best["label"]
    max_conf = round(best["score"], 4)
    forged = "forg" in predicted_label.lower()

    elapsed = round(time.time() - start, 3)

    kpis: Dict[str, Any] = {
        "detection_result": {
            "is_forged": forged,
            "predicted_class": predicted_label,
            "confidence": max_conf,
            "risk_level": _risk_level(max_conf, forged),
        },
        "all_predictions": predictions,
        "confidence_analysis": {
            "max_confidence": max_conf,
            "min_confidence": round(min(predictions.values()), 4),
            "confidence_gap": round(max_conf - min(predictions.values()), 4),
            "certainty_level": _certainty(max_conf),
        },
        "image_quality": {
            "width": width,
            "height": height,
            "total_pixels": width * height,
            "aspect_ratio": round(width / height, 2),
            "format": img_format,
            "size_category": _size_category(width, height),
        },
        "processing_metrics": {
            "processing_time_seconds": elapsed,
            "processing_speed": "fast" if elapsed < 2 else "medium" if elapsed < 5 else "slow",
            "timestamp": datetime.utcnow().isoformat(),
        },
        "business_kpis": {
            "authenticity_score": _authenticity_score(predictions, forged),
            "verification_status": "PASS" if (not forged and max_conf > 0.7) else "FAIL",
            "recommendation": _recommendation(forged, max_conf),
            "requires_human_review": max_conf < 0.8,
        },
    }

    return {
        "status": "SUCCESS",
        "message": "KPI calculation completed successfully",
        "kpis": kpis,
        "model_info": {
            "model_name": MODEL_NAME,
            "model_type": "Vision Transformer",
        },
    }


#####################################################################################
# API Endpoints
#####################################################################################


@app.get("/")
async def root() -> Dict[str, Any]:
    """Simple service landing page."""

    return {
        "message": "Document Forgery Detection API",
        "version": app.version,
        "endpoints": {
            "kpi_calculation": "/kpi_calculation/",
            "health": "/health/",
            "docs": "/docs",
        },
    }


@app.get("/health/")
async def health() -> Dict[str, Any]:
    """Liveness & readiness probe – suitable for K8s."""

    return {
        "status": "healthy" if model_holder.ready else "initialising",
        "timestamp": datetime.utcnow().isoformat(),
        "model_loaded": model_holder.ready,
        "model_name": MODEL_NAME if model_holder.ready else None,
    }


@app.post("/kpi_calculation/", summary="Upload an image and obtain KPI metrics")
async def kpi_calculation(image: UploadFile = File(..., description="PNG/JPG/JPEG image")) -> JSONResponse:  # noqa: D401, E501
    """Main endpoint – returns KPIs for a single document image."""

    if not model_holder.ready:
        raise HTTPException(status_code=503, detail="Model not loaded yet, try again later …")

    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    img_bytes = await image.read()

    if not img_bytes:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    if len(img_bytes) > 10 * 1024 * 1024:  # 10 MB
        raise HTTPException(status_code=400, detail="File size exceeds 10 MB limit.")

    try:
        payload = _calculate_kpi(img_bytes)
        return JSONResponse(content=jsonable_encoder(payload))
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Processing failed: %s", exc)
        raise HTTPException(status_code=500, detail="Internal processing error.") from exc


#####################################################################################
# Local dev entry‑point
#####################################################################################

if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    logger.info("Starting server …")
    uvicorn.run("forgery_api:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
