"""HTTP layer (FastAPI) for the voice-clone detector.

All inference logic lives in vcd_service / dhwani_infer; this file only handles
auth, upload limits, temp files and status codes. Model: third-party Dhwani
ONNX checkpoint (MIT), see vcd_service.ATTRIBUTION.

Environment:
  VCD_API_KEY            required, >= 16 chars; sent by clients as X-API-Key
  VCD_MODEL_PATH         pinned checkpoint path (else local HF cache)
  VCD_ALLOW_DOWNLOAD     "1" to allow one download at startup (default off)
  VCD_PREPROCESSING      default mode (train_peak | train_zscore | api_replica)
  VCD_MAX_UPLOAD_BYTES   default 8 MiB
  VCD_TMP_DIR            where uploads are written briefly (deleted after each request)
  VCD_MAX_WAIT_S         max seconds a request waits for the inference slot (default 60)

Run (single worker - one model, one inference at a time):
  uvicorn vcd_api:app --app-dir <dir with these files> --host 127.0.0.1 --port 8000 --workers 1
"""

from __future__ import annotations

import hmac
import logging
import os
import tempfile
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse

from vcd_service import DetectorService, ServiceBusy, ServiceNotReady

API_KEY = os.environ.get("VCD_API_KEY", "")
MAX_UPLOAD_BYTES = int(os.environ.get("VCD_MAX_UPLOAD_BYTES", str(8 * 1024 * 1024)))
MULTIPART_OVERHEAD = 64 * 1024
TMP_DIR = os.environ.get("VCD_TMP_DIR") or os.path.join(tempfile.gettempdir(), "vcd_api_tmp")
ALLOWED_EXT = {".wav", ".m4a", ".mp3", ".aac", ".ogg", ".opus", ".flac", ".3gp", ".amr", ".webm", ".mp4"}

log = logging.getLogger("vcd_api")
service = DetectorService(
    default_mode=os.environ.get("VCD_PREPROCESSING", "train_peak"),
    allow_download=os.environ.get("VCD_ALLOW_DOWNLOAD") == "1",
    max_wait_s=float(os.environ.get("VCD_MAX_WAIT_S", "60")),
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if len(API_KEY) < 16:
        raise RuntimeError("set VCD_API_KEY (at least 16 characters) before starting the API")
    os.makedirs(TMP_DIR, exist_ok=True)
    service.load()  # once per process; a failure is reported by /health
    log.info("model ready=%s error=%s", service.ready, service.load_error)
    yield


app = FastAPI(title="Voice-clone detector (Dhwani baseline)", version="0.1.0", lifespan=lifespan)


class BodySizeLimit:
    """Reject oversized request bodies before they are spooled to disk."""

    def __init__(self, app, max_bytes: int):
        self.app, self.max_bytes = app, max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        cl = dict(scope.get("headers") or []).get(b"content-length")
        if cl is not None and cl.isdigit() and int(cl) > self.max_bytes:
            resp = JSONResponse(status_code=413, content=_err("upload_too_large",
                                f"request body exceeds {self.max_bytes} bytes"))
            return await resp(scope, receive, send)
        seen = 0

        async def limited_receive():
            nonlocal seen
            msg = await receive()
            if msg["type"] == "http.request":
                seen += len(msg.get("body", b""))
                if seen > self.max_bytes:  # chunked uploads without Content-Length
                    raise HTTPException(413, _err("upload_too_large", f"request body exceeds {self.max_bytes} bytes"))
            return msg

        await self.app(scope, limited_receive, send)


app.add_middleware(BodySizeLimit, max_bytes=MAX_UPLOAD_BYTES + MULTIPART_OVERHEAD)


def _err(code: str, message: str) -> dict:
    return {"status": "error", "error": code, "message": message}


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if not x_api_key or not hmac.compare_digest(x_api_key.encode(), API_KEY.encode()):
        raise HTTPException(401, _err("unauthorized", "missing or invalid X-API-Key header"))


@app.exception_handler(HTTPException)
async def _http_error(request, exc: HTTPException):
    body = exc.detail if isinstance(exc.detail, dict) else _err("http_error", str(exc.detail))
    return JSONResponse(status_code=exc.status_code, content=body, headers=getattr(exc, "headers", None))


@app.get("/health")
def health():
    return service.health()


@app.post("/detect", dependencies=[Depends(require_api_key)])
def detect(file: UploadFile = File(...), mode: str | None = Query(default=None)):
    t0 = time.perf_counter()
    request_id = uuid.uuid4().hex[:12]
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(415, _err("unsupported_type", f"extension {ext or '(none)'} not in {sorted(ALLOWED_EXT)}"))
    if not service.ready:
        raise HTTPException(503, _err("model_not_ready", service.load_error or "model not loaded"))

    fd, tmp_path = tempfile.mkstemp(suffix=ext, dir=TMP_DIR)
    try:
        written = 0
        with os.fdopen(fd, "wb") as out:
            while chunk := file.file.read(1 << 20):
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, _err("upload_too_large", f"file exceeds {MAX_UPLOAD_BYTES} bytes"))
                out.write(chunk)
        if written == 0:
            raise HTTPException(422, {**_err("rejected", "empty upload (0 bytes)"),
                                      "status": "rejected", "reject_reason": "empty upload (0 bytes)"})
        try:
            result = service.detect_file(tmp_path, mode=mode)
        except ValueError as e:
            raise HTTPException(400, _err("bad_preprocessing_mode", str(e)))
        except ServiceBusy as e:
            raise HTTPException(503, _err("busy", str(e)), headers={"Retry-After": "5"})
        except ServiceNotReady as e:
            raise HTTPException(503, _err("model_not_ready", str(e)))
    finally:
        file.file.close()
        try:
            os.remove(tmp_path)
        except FileNotFoundError:
            pass

    result["request_id"] = request_id
    result["upload_bytes"] = written
    result["timing_ms"]["total"] = round((time.perf_counter() - t0) * 1000, 1)
    log.info("detect %s status=%s mode=%s total_ms=%s", request_id, result["status"],
             result["preprocessing"], result["timing_ms"]["total"])
    return JSONResponse(status_code=200 if result["status"] == "ok" else 422, content=result)
