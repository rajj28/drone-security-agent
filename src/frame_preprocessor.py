"""
frame_preprocessor.py — Image quality assessment and enhancement for surveillance frames.

Shared by vision_analyzer and cloud_enhanced_analyzer to improve VLM accuracy on
blurry, dark, or low-contrast drone footage without requiring extra API calls.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from src.config import settings

logger = logging.getLogger(__name__)

BLUR_THRESHOLD = 100.0
DARK_THRESHOLD = 50.0
LOW_CONTRAST_THRESHOLD = 30.0
NOISE_THRESHOLD = 18.0


def preprocessing_enabled() -> bool:
    """Whether robust frame preprocessing is active (ROBUST_PREPROCESS env)."""
    return bool(settings.ROBUST_PREPROCESS)


def assess_image_quality(image_path: Path) -> Dict[str, float]:
    """Return blur, brightness, contrast, and noise metrics for a frame."""
    img_array = np.array(Image.open(image_path).convert("RGB"))
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    denoised = cv2.GaussianBlur(gray, (5, 5), 0)
    noise_score = float(np.std(gray.astype(np.float32) - denoised.astype(np.float32)))
    quality_score = _overall_quality_score(blur_score, brightness, contrast, noise_score)
    return {
        "blur_score": blur_score,
        "brightness": brightness,
        "contrast": contrast,
        "noise_score": noise_score,
        "quality_score": quality_score,
    }


def _overall_quality_score(blur: float, brightness: float, contrast: float, noise: float) -> float:
    """0–100 composite quality score (higher = better for VLM analysis)."""
    blur_norm = min(blur / 200.0, 1.0) * 35
    bright_norm = (1.0 - abs(brightness - 128.0) / 128.0) * 25
    contrast_norm = min(contrast / 60.0, 1.0) * 25
    noise_norm = max(0.0, 1.0 - noise / 40.0) * 15
    return round(blur_norm + bright_norm + contrast_norm + noise_norm, 1)


def _apply_clahe_rgb(img_array: np.ndarray) -> np.ndarray:
    """Apply CLAHE on the L channel in LAB space — good for dark CCTV footage."""
    lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_channel)
    merged = cv2.merge([l_enhanced, a_channel, b_channel])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)


def _denoise_if_needed(img_array: np.ndarray, noise_score: float) -> np.ndarray:
    if noise_score < NOISE_THRESHOLD:
        return img_array
    return cv2.fastNlMeansDenoisingColored(img_array, None, 6, 6, 7, 21)


def preprocess_frame_image(
    image_path: Path,
    *,
    jpeg_quality: int = 90,
    force: bool = False,
) -> Tuple[bytes, Dict[str, float]]:
    """
    Enhance frame when quality is poor and return JPEG bytes plus quality metrics.
    Falls back to the original file bytes if preprocessing fails or is disabled.
    """
    quality = assess_image_quality(image_path)
    if not force and not preprocessing_enabled():
        quality["preprocessed"] = 0.0
        return image_path.read_bytes(), quality

    try:
        img = Image.open(image_path)
        if img.mode != "RGB":
            img = img.convert("RGB")

        enhanced = False
        img_array = np.array(img)

        if quality["noise_score"] >= NOISE_THRESHOLD:
            img_array = _denoise_if_needed(img_array, quality["noise_score"])
            enhanced = True

        if quality["blur_score"] < BLUR_THRESHOLD:
            img = Image.fromarray(img_array).filter(ImageFilter.SHARPEN)
            img_array = np.array(img)
            enhanced = True

        if quality["brightness"] < DARK_THRESHOLD:
            img_array = _apply_clahe_rgb(img_array)
            img = Image.fromarray(img_array)
            img = ImageEnhance.Brightness(img).enhance(1.2)
            img_array = np.array(img)
            enhanced = True

        if quality["contrast"] < LOW_CONTRAST_THRESHOLD:
            img = Image.fromarray(img_array)
            img = ImageEnhance.Contrast(img).enhance(1.4)
            img_array = np.array(img)
            enhanced = True

        img = Image.fromarray(img_array)
        if img.size[0] > 1920 or img.size[1] > 1080:
            img.thumbnail((1920, 1080), Image.Resampling.LANCZOS)
            enhanced = True

        if enhanced:
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
            quality["preprocessed"] = 1.0
            logger.info(
                "Preprocessed %s (blur=%.1f brightness=%.1f contrast=%.1f quality=%.1f)",
                image_path.name,
                quality["blur_score"],
                quality["brightness"],
                quality["contrast"],
                quality["quality_score"],
            )
            return buf.getvalue(), quality

        quality["preprocessed"] = 0.0
        return image_path.read_bytes(), quality
    except Exception as exc:
        logger.warning("Frame preprocessing failed for %s: %s", image_path, exc)
        quality["preprocessed"] = 0.0
        return image_path.read_bytes(), quality


def quality_prompt_hints(metrics: Dict[str, Any]) -> str:
    """Build optional VLM prompt hints from quality metrics."""
    hints = []
    if metrics.get("blur_score", 999) < BLUR_THRESHOLD:
        hints.append("Image is blurry — describe only clearly visible elements and note uncertainty.")
    if metrics.get("brightness", 128) < DARK_THRESHOLD:
        hints.append("Image is dark — pay attention to silhouettes, reflections, and light sources.")
    if metrics.get("contrast", 50) < LOW_CONTRAST_THRESHOLD:
        hints.append("Image has low contrast — be conservative in threat assessment.")
    if metrics.get("noise_score", 0) >= NOISE_THRESHOLD:
        hints.append("Image is noisy/grainy — ignore compression artifacts when assessing threats.")
    if metrics.get("quality_score", 100) < 40:
        hints.append("Overall image quality is poor — lower confidence on fine details like faces or small objects.")
    if not hints:
        return ""
    return "\n\n=== IMAGE QUALITY NOTES ===\n" + " ".join(hints)
