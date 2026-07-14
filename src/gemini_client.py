"""
gemini_client.py — Google Gemini API (vision + text + embeddings).

Model order controlled by GEMINI_PREFER_FLASH (default True for production).
"""

from __future__ import annotations

import base64
import json
import os
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import urllib.error
import urllib.request

from src.api_retry import call_with_retry, configure_min_interval, is_quota_exhausted_error
from src.config import settings

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"


def _load_api_keys() -> List[str]:
    """Collect all configured Gemini API keys (primary + extras), de-duplicated, in order."""
    candidates = [
        settings.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY", ""),
        getattr(settings, "GEMINI_API_KEY_2", "") or os.environ.get("GEMINI_API_KEY_2", ""),
        getattr(settings, "GEMINI_API_KEY_3", "") or os.environ.get("GEMINI_API_KEY_3", ""),
    ]
    keys: List[str] = []
    for raw in candidates:
        key = (raw or "").strip()
        if key and key not in keys:
            keys.append(key)
    return keys


_API_KEYS: List[str] = _load_api_keys()
_key_lock = threading.Lock()
_key_idx = 0

# Scale the global call spacing by the number of keys so adding keys raises throughput,
# while each key still respects the per-key interval (API_MIN_INTERVAL_SEC).
try:
    if getattr(settings, "USE_VERTEX_AI", False):
        interval = 2.5
    else:
        interval = float(settings.API_MIN_INTERVAL_SEC)
    configure_min_interval(interval, max(1, len(_API_KEYS)))
    if len(_API_KEYS) > 1:
        print(f"[GEMINI] Load-sharing across {len(_API_KEYS)} API keys (round-robin)")
except Exception:
    pass


def _api_key() -> str:
    """Return the next API key (round-robin across all configured keys), or an ADC token."""
    global _key_idx
    if _API_KEYS:
        with _key_lock:
            key = _API_KEYS[_key_idx % len(_API_KEYS)]
            _key_idx += 1
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


def _get_vertex_token() -> str:
    """Acquire an OAuth access token for Vertex AI."""
    # 1. Check if GCP_ADC_JSON is set in the environment (used in production / Fly.io)
    gcp_adc_json = os.environ.get("GCP_ADC_JSON")
    if gcp_adc_json:
        try:
            import google.auth
            import google.auth.transport.requests
            
            # If base64 encoded, decode it
            if not gcp_adc_json.strip().startswith("{"):
                import base64
                gcp_adc_json = base64.b64decode(gcp_adc_json).decode("utf-8")
                
            info = json.loads(gcp_adc_json)
            if info.get("type") == "service_account":
                from google.oauth2.service_account import Credentials as ServiceAccountCredentials
                creds = ServiceAccountCredentials.from_service_account_info(
                    info, scopes=["https://www.googleapis.com/auth/cloud-platform"]
                )
            else:
                from google.oauth2.credentials import Credentials as UserCredentials
                creds = UserCredentials.from_authorized_user_info(info)
                
            auth_req = google.auth.transport.requests.Request()
            creds.refresh(auth_req)
            if creds.token:
                return creds.token
        except Exception as e:
            print(f"[VERTEX] Failed to load credentials from GCP_ADC_JSON: {e}")

    # 2. Try standard google.auth
    try:
        import google.auth
        import google.auth.transport.requests
        creds, _ = google.auth.default()
        auth_req = google.auth.transport.requests.Request()
        creds.refresh(auth_req)
        if creds.token:
            return creds.token
    except Exception:
        pass

    # 3. Fall back to gcloud CLI (local development)
    import subprocess
    try:
        return subprocess.check_output("gcloud auth print-access-token", shell=True).decode("utf-8").strip()
    except Exception as e:
        raise RuntimeError("No Google Cloud credentials found for Vertex AI. Run 'gcloud auth login'") from e


def _post_once(path: str, body: Dict[str, Any]) -> Dict[str, Any]:
    if getattr(settings, "USE_VERTEX_AI", False):
        token = _get_vertex_token()
        project = settings.GCP_PROJECT_ID
        location = settings.GCP_LOCATION
        # path is like "models/gemini-2.5-flash:generateContent" or "models/text-embedding-004:embedContent"
        # We need to extract the model name and action
        # e.g., models/gemini-2.5-flash:generateContent -> gemini-2.5-flash and generateContent
        match = re.search(r"models/([^:]+):(.+)", path)
        if match:
            model_name = match.group(1)
            action = match.group(2)
        else:
            model_name = "gemini-2.5-flash"
            action = "generateContent"

        url = f"https://{location}-aiplatform.googleapis.com/v1/projects/{project}/locations/{location}/publishers/google/models/{model_name}:{action}"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        }
        # Vertex AI requires "role": "user" or "model" in the content structures
        if "contents" in body:
            new_contents = []
            for item in body["contents"]:
                if "parts" in item and "role" not in item:
                    item["role"] = "user"
                new_contents.append(item)
            body["contents"] = new_contents
    else:
        creds = _api_key()
        # ADC tokens use a Bearer Authorization header.
        # API keys (both legacy "AIza" standard keys and new "AQ." auth keys) are sent via
        # the x-goog-api-key header. The legacy ?key= query param does NOT work for AQ. keys
        # (returns 401), so we always use the header form.
        if creds.startswith("ADC:"):
            token = creds[4:]  # Strip marker
            url = f"{GEMINI_API_BASE}/{path}"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}",
            }
        else:
            url = f"{GEMINI_API_BASE}/{path}"
            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": creds,
            }
    
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        retry_after = _parse_retry_after(raw)
        raise GeminiApiError(exc.code, f"[{exc.code}] {raw[:500]}", retry_after) from exc


def _post(path: str, body: Dict[str, Any]) -> Dict[str, Any]:
    """Send a request. For standard API keys, support immediate key failover.
    For Vertex AI, make calls directly without key rotation."""
    if getattr(settings, "USE_VERTEX_AI", False):
        return _post_once(path, body)

    attempts = max(1, len(_API_KEYS))
    last_exc: Optional[BaseException] = None
    for i in range(attempts):
        try:
            return _post_once(path, body)
        except GeminiApiError as exc:
            last_exc = exc
            failover_worthy = exc.status in (429, 503, 500)
            if failover_worthy and i < attempts - 1 and len(_API_KEYS) > 1:
                print(f"[GEMINI] Key rate-limited ({exc.status}); failing over to next key ({i + 2}/{attempts})")
                continue
            raise
    if last_exc:
        raise last_exc
    raise RuntimeError("Gemini request failed with no response")


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
