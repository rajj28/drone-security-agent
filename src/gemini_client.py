"""
gemini_client.py — Google Gemini API (vision + text + embeddings).

Model order controlled by GEMINI_PREFER_FLASH (default True for production).
"""

from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import urllib.error
import urllib.request

from src.api_retry import call_with_retry, is_quota_exhausted_error
from src.config import settings

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"


def _api_key() -> str:
    """Get API key from settings or ADC."""
    key = settings.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY", "")
    if key:
        return key
    # Try Application Default Credentials (ADC) for orgs that block API keys
    try:
        import google.auth
        import google.auth.transport.requests
        credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/generative-language"])
        credentials.refresh(google.auth.transport.requests.Request())
        if credentials.token:
            return f"ADC:{credentials.token}"  # Marker for ADC token
    except Exception:
        pass
    raise ValueError("GEMINI_API_KEY is not set and ADC failed")


class GeminiApiError(RuntimeError):
    def __init__(self, status: int, message: str, retry_after_sec: float | None = None):
        super().__init__(message)
        self.status = status
        self.retry_after_sec = retry_after_sec


def _parse_retry_after(body: str) -> float | None:
    match = re.search(r"retry in ([0-9.]+)s", body, re.I)
    if match:
        return float(match.group(1))
    match = re.search(r'"retryDelay"\s*:\s*"([0-9]+)s"', body)
    if match:
        return float(match.group(1))
    return None


def _post(path: str, body: Dict[str, Any]) -> Dict[str, Any]:
    creds = _api_key()
    # ADC tokens use Authorization header; API keys use URL param
    if creds.startswith("ADC:"):
        token = creds[4:]  # Strip marker
        url = f"{GEMINI_API_BASE}/{path}"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        }
    else:
        url = f"{GEMINI_API_BASE}/{path}?key={creds}"
        headers = {"Content-Type": "application/json"}
    
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        retry_after = _parse_retry_after(raw)
        raise GeminiApiError(exc.code, f"[{exc.code}] {raw[:500]}", retry_after) from exc


def _extract_text(response: Dict[str, Any]) -> str:
    candidates = response.get("candidates") or []
    if not candidates:
        raise RuntimeError(f"Gemini returned no candidates: {response}")
    parts = candidates[0].get("content", {}).get("parts") or []
    texts = [p.get("text", "") for p in parts if p.get("text")]
    if not texts:
        raise RuntimeError(f"Gemini returned empty text: {response}")
    return "\n".join(texts)


def _is_model_unavailable_error(exc: BaseException) -> bool:
    if isinstance(exc, GeminiApiError) and exc.status in (429, 503, 500):
        return True
    text = str(exc).lower()
    return (
        "429" in text
        or "quota" in text
        or "resource_exhausted" in text
        or "rate limit" in text
        or "503" in text
    )


def _model_chain(explicit: Optional[str] = None) -> List[str]:
    """Order models for this call; dedupe while preserving order."""
    if settings.GEMINI_FLASH_ONLY and not explicit:
        return [settings.GEMINI_FALLBACK_MODEL]
    if explicit:
        primary, secondary = explicit, None
    elif settings.GEMINI_PREFER_FLASH:
        primary = settings.GEMINI_FALLBACK_MODEL
        secondary = settings.GEMINI_MODEL
    else:
        primary = settings.GEMINI_MODEL
        secondary = settings.GEMINI_FALLBACK_MODEL

    out: List[str] = []
    for name in (primary, secondary):
        if name and name not in out:
            out.append(name)
    return out


def generate_text(
    prompt: str,
    *,
    system_instruction: Optional[str] = None,
    model: Optional[str] = None,
    max_output_tokens: int = 2048,
    temperature: float = 0.2,
) -> str:
    """Text generation with configurable model order and rate-limit aware retries."""
    models = _model_chain(model)
    last_error: Optional[Exception] = None

    for model_name in models:
        body: Dict[str, Any] = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_output_tokens,
            },
        }
        if system_instruction:
            body["systemInstruction"] = {"parts": [{"text": system_instruction}]}

        def _call(m=model_name):
            return _post(f"models/{m}:generateContent", body)

        try:
            # Fewer retries on primary when Flash-first — fail over to next model quickly
            retries = 2 if settings.GEMINI_PREFER_FLASH and model_name == models[0] else 4
            return _extract_text(
                call_with_retry(_call, label=f"gemini-{model_name}", max_retries=retries)
            )
        except Exception as exc:
            last_error = exc
            if _is_model_unavailable_error(exc) and model_name != models[-1]:
                print(f"[GEMINI] {model_name} unavailable, trying next model")
                continue
            raise

    raise last_error or RuntimeError("Gemini text generation failed")


def generate_vision(
    prompt: str,
    image: Union[Path, str, bytes],
    *,
    mime_type: str = "image/jpeg",
    model: Optional[str] = None,
    max_output_tokens: int = 2048,
    temperature: float = 0.2,
) -> Dict[str, Any]:
    """Vision + text generation."""
    if isinstance(image, Path):
        raw = image.read_bytes()
    elif isinstance(image, bytes):
        raw = image
    else:
        raw = Path(image).read_bytes()

    b64 = base64.b64encode(raw).decode("ascii")
    models = _model_chain(model)
    last_error: Optional[Exception] = None

    for model_name in models:
        body = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {"inline_data": {"mime_type": mime_type, "data": b64}},
                    ]
                }
            ],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_output_tokens,
            },
        }

        def _call(m=model_name):
            return _post(f"models/{m}:generateContent", body)

        try:
            retries = 2 if settings.GEMINI_PREFER_FLASH and model_name == models[0] else 4
            text = _extract_text(
                call_with_retry(
                    _call, label=f"gemini-vision-{model_name}", max_retries=retries
                )
            )
            return {
                "success": True,
                "text": text,
                "model_used": model_name,
            }
        except Exception as exc:
            last_error = exc
            if _is_model_unavailable_error(exc) and model_name != models[-1]:
                print(f"[GEMINI] Vision {model_name} unavailable, trying next model")
                continue
            return {"success": False, "error": str(exc), "text": "", "model_used": model_name}

    if last_error and is_quota_exhausted_error(last_error):
        from src.api_retry import mark_quota_exhausted

        mark_quota_exhausted()
    return {
        "success": False,
        "error": str(last_error or "unknown"),
        "text": "",
        "model_used": models[0] if models else "unknown",
    }


def embed_text(text: str) -> List[float]:
    """Gemini embedding for Pinecone / search (legacy BYOV mode)."""
    model = settings.GEMINI_EMBEDDING_MODEL
    body: Dict[str, Any] = {
        "content": {"parts": [{"text": text[:8000]}]},
    }
    dim = settings.GEMINI_EMBEDDING_DIMENSION
    if dim:
        body["outputDimensionality"] = dim

    def _call():
        return _post(f"models/{model}:embedContent", body)

    result = call_with_retry(_call, label="gemini-embed")
    values = result.get("embedding", {}).get("values")
    if not values:
        raise RuntimeError(f"Gemini embedding failed: {result}")
    return values
