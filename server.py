"""
Ζωγραφιά με Ζωή AI — backend.

What it does:
  POST /api/animate-drawing   → analyzes a kid's drawing with OpenAI Vision
                                 and returns a short Greek story + speakable
                                 dialogue lines that "bring the drawing to life".
                                 Counts as 1 "animation use".

  POST /api/tts                → mp3 audio (OpenAI TTS, default voice "shimmer")
                                 — used for "play story" + "play again".
                                 Not gated (cheap, core UX).

  GET  /api/usage              → device's plan + remaining animations.

  POST /api/checkout           → Stripe Checkout subscription session.

  POST /api/stripe-webhook     → activates/cancels the device's subscription.

Plans:
  trial  — 3 free animations.
  basic  — €2.99/month, 15 animations/month, refills with billing period.
  full   — €5.99/month or €49.99/year, unlimited.
"""

import base64
import io
import json
import os
import re
import secrets
import sqlite3
import time
from contextlib import contextmanager
from typing import Optional

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from openai import OpenAI
from pydantic import BaseModel

# --- Config -----------------------------------------------------------------
DB_PATH = os.environ.get("DB_PATH", "/var/data/zografia.db")
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "/var/data/uploads")

# Quotas tuned for the cheap Akool Basic Fast model (~4 cr/video) and
# AKOOL's credit pricing (~$0.03/cr ≈ $0.12/video + ~$0.005 OpenAI).
# Margins per plan (€1 ≈ $1.07, prices below assume €):
#   Basic  10 vids × $0.13 = $1.30 cost  vs €2.99 (~$3.20)  → ~$1.90 margin
#   Full   25 vids × $0.13 = $3.25 cost  vs €5.99 (~$6.40)  → ~$3.15 margin
#   Yearly 300 vids × $0.13 = $39 cost   vs €49.99 (~$53)   → ~$14 margin
TRIAL_LIMIT  = 3
BASIC_LIMIT  = 10
FULL_LIMIT   = 25
YEARLY_LIMIT = 300

AKOOL_API_KEY      = os.environ.get("AKOOL_API_KEY", "").strip()
AKOOL_BASE         = "https://openapi.akool.com/api/open/v3"          # legacy
AKOOL_I2V_BASE     = "https://openapi.akool.com/api/open/v4/image2Video"  # image-to-video
AKOOL_RESOLUTION   = os.environ.get("AKOOL_RESOLUTION", "720p").strip() or "720p"
AKOOL_VIDEO_LENGTH = int(os.environ.get("AKOOL_VIDEO_LENGTH", "5") or "5")
# Cheap Akool Basic model = 4 credits/video at 720p. Premium models (Seedance,
# MiniMax, Sora, Kling) cost ~100. Override via env if quality matters more.
AKOOL_MODEL_NAME   = os.environ.get("AKOOL_MODEL_NAME", "AkoolImage2VideoFastV1").strip() \
                     or "AkoolImage2VideoFastV1"

# Owners (events/presentations) get unlimited usage. Comma-separated list of
# lowercased emails. The user enters their email via /api/owner-unlock and if
# it matches one in this set, their device's plan is upgraded to 'owner'.
OWNER_EMAILS = {
    e.strip().lower() for e in os.environ.get(
        "OWNER_EMAILS",
        "events.topolytexno@gmail.com,info@evlabsai.gr,evaggelos77@gmail.com"
    ).split(",")
    if e.strip()
}
PUBLIC_BASE_URL = os.environ.get(
    "PUBLIC_BASE_URL", "https://zografia-ai.onrender.com"
).rstrip("/")

PRICE_BASIC_MONTHLY = os.environ.get("PRICE_BASIC_MONTHLY", "").strip()
PRICE_FULL_MONTHLY  = os.environ.get("PRICE_FULL_MONTHLY", "").strip()
PRICE_FULL_YEARLY   = os.environ.get("PRICE_FULL_YEARLY", "").strip()
STRIPE_SECRET_KEY   = os.environ.get("STRIPE_SECRET_KEY", "").strip()
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "").strip()

# --- App + CORS -------------------------------------------------------------
app = FastAPI(title="Ζωγραφιά με Ζωή AI backend", version="1.0.0")

ALLOWED_ORIGINS = [
    "https://zografia-ai.onrender.com",
    "https://zografia.onrender.com",
    "https://evlabsai.gr",
    "https://www.evlabsai.gr",
    "http://localhost:3000",
    "http://localhost:5500",
    "http://localhost:8787",
    "http://127.0.0.1:5500",
    "http://127.0.0.1:8787",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Device-Id"],
)

# --- SQLite -----------------------------------------------------------------
def _init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with sqlite3.connect(DB_PATH) as con:
        con.executescript(
            """
            CREATE TABLE IF NOT EXISTS devices (
                id                     TEXT PRIMARY KEY,
                plan                   TEXT NOT NULL DEFAULT 'trial',
                plan_status            TEXT NOT NULL DEFAULT 'active',
                uses                   INTEGER NOT NULL DEFAULT 0,
                period_start           INTEGER NOT NULL DEFAULT 0,
                current_period_end     INTEGER NOT NULL DEFAULT 0,
                stripe_customer_id     TEXT DEFAULT '',
                stripe_subscription_id TEXT DEFAULT '',
                created_at             INTEGER NOT NULL,
                updated_at             INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_devices_sub ON devices(stripe_subscription_id);
            CREATE INDEX IF NOT EXISTS idx_devices_cus ON devices(stripe_customer_id);
            """
        )
_init_db()

# Ensure upload dir exists (we save resized drawings here so AKOOL can fetch them)
os.makedirs(UPLOAD_DIR, exist_ok=True)


@contextmanager
def db():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()


def _now() -> int:
    return int(time.time())


def get_or_create_device(device_id: str) -> dict:
    if not device_id or len(device_id) < 8 or len(device_id) > 80:
        raise HTTPException(status_code=400, detail="Invalid device id")
    now = _now()
    with db() as con:
        row = con.execute("SELECT * FROM devices WHERE id=?", (device_id,)).fetchone()
        if row:
            return dict(row)
        con.execute(
            "INSERT INTO devices (id, created_at, updated_at) VALUES (?,?,?)",
            (device_id, now, now),
        )
        row = con.execute("SELECT * FROM devices WHERE id=?", (device_id,)).fetchone()
        return dict(row)


def _refresh_billing_period(dev: dict) -> dict:
    if dev.get("plan") == "basic" and dev.get("period_start"):
        cpe = int(dev.get("current_period_end") or 0)
        if cpe and _now() >= cpe:
            with db() as con:
                con.execute(
                    "UPDATE devices SET uses=0, period_start=?, updated_at=? WHERE id=?",
                    (_now(), _now(), dev["id"]),
                )
            dev["uses"] = 0
            dev["period_start"] = _now()
    return dev


def _quota_for(plan: str) -> Optional[int]:
    if plan == "owner":  return None     # unlimited (events/presentations)
    if plan == "yearly": return YEARLY_LIMIT
    if plan == "full":   return FULL_LIMIT
    if plan == "basic":  return BASIC_LIMIT
    return TRIAL_LIMIT


def _plan_status_dict(dev: dict) -> dict:
    plan = dev.get("plan") or "trial"
    status = dev.get("plan_status") or "active"
    is_active = status == "active"
    is_paid = plan in ("basic", "full", "yearly", "owner") and is_active
    quota = _quota_for(plan if is_active else "trial")
    used = int(dev.get("uses") or 0)
    remaining = max(0, quota - used) if quota is not None else None
    return {
        "device_id": dev["id"],
        "plan": plan,
        "plan_status": status,
        "is_paid": is_paid,
        "is_active": is_active,
        "used": used,
        "quota": quota,
        "remaining": remaining,
        "current_period_end": int(dev.get("current_period_end") or 0),
    }


def _consume(device_id: str) -> dict:
    with db() as con:
        row = con.execute("SELECT * FROM devices WHERE id=?", (device_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=400, detail="Unknown device")
        dev = dict(row)
        dev = _refresh_billing_period(dev)
        plan = dev.get("plan") or "trial"
        is_active = (dev.get("plan_status") or "active") == "active"
        effective = plan if is_active else "trial"
        quota = _quota_for(effective)
        used = int(dev.get("uses") or 0)
        if quota is not None and used >= quota:
            raise HTTPException(status_code=402, detail="quota_exceeded")
        con.execute(
            "UPDATE devices SET uses=?, updated_at=? WHERE id=?",
            (used + 1, _now(), device_id),
        )
        dev["uses"] = used + 1
        return dev


def _require_device(x_device_id: Optional[str]) -> dict:
    if not x_device_id:
        raise HTTPException(status_code=400, detail="X-Device-Id header required")
    return get_or_create_device(x_device_id)


def _refund(device_id: str):
    try:
        with db() as con:
            con.execute(
                "UPDATE devices SET uses=MAX(0, uses-1), updated_at=? WHERE id=?",
                (_now(), device_id),
            )
    except Exception:
        pass


# --- OpenAI client ----------------------------------------------------------
_oa: Optional[OpenAI] = None


def openai_client() -> OpenAI:
    global _oa
    if _oa is None:
        key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not key:
            raise HTTPException(status_code=503, detail="OpenAI not configured")
        _oa = OpenAI(api_key=key)
    return _oa


# --- Stripe client ----------------------------------------------------------
_stripe = None


def stripe_module():
    global _stripe
    if _stripe is None:
        if not STRIPE_SECRET_KEY:
            raise HTTPException(status_code=503, detail="Stripe not configured")
        import stripe as _s
        _s.api_key = STRIPE_SECRET_KEY
        _stripe = _s
    return _stripe


PLAN_FROM_PRICE = {
    PRICE_BASIC_MONTHLY: "basic",
    PRICE_FULL_MONTHLY:  "full",
    PRICE_FULL_YEARLY:   "yearly",
}


# ============================================================================
# Routes
# ============================================================================
# --- AKOOL Talking-Photo + file hosting -------------------------------------
ALLOWED_UPLOAD_EXT = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
                     "mp3": "audio/mpeg", "wav": "audio/wav"}


def _save_bytes(data: bytes, ext: str) -> str:
    """Save bytes under UPLOAD_DIR/<random>.<ext>. Returns the filename (with ext)."""
    ext = ext.strip(".").lower()
    if ext not in ALLOWED_UPLOAD_EXT:
        ext = "bin"
    if len(data) > 8 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="file too large")
    name = secrets.token_urlsafe(18)
    with open(os.path.join(UPLOAD_DIR, f"{name}.{ext}"), "wb") as f:
        f.write(data)
    return f"{name}.{ext}"


def _save_image_to_disk(data_url: str) -> str:
    if not data_url or ";base64," not in data_url:
        raise HTTPException(status_code=400, detail="image data URL required")
    b64 = data_url.split(";base64,", 1)[1]
    try:
        raw = base64.b64decode(b64)
    except Exception:
        raise HTTPException(status_code=400, detail="image not base64-decodable")
    return _save_bytes(raw, "jpg")


@app.get("/uploads/{filename}")
def get_upload(filename: str):
    safe = re.sub(r"[^A-Za-z0-9._-]", "", filename)[:80]
    if not safe or "/" in safe or ".." in safe:
        raise HTTPException(status_code=404, detail="not found")
    path = os.path.join(UPLOAD_DIR, safe)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="not found")
    ext = safe.rsplit(".", 1)[-1].lower() if "." in safe else ""
    mime = ALLOWED_UPLOAD_EXT.get(ext, "application/octet-stream")
    return FileResponse(path, media_type=mime, headers={"Cache-Control": "public, max-age=86400"})


def _backend_origin() -> str:
    """Public origin for our uploads/* files (AKOOL needs HTTPS URLs)."""
    explicit = os.environ.get("BACKEND_PUBLIC_URL", "").rstrip("/")
    if explicit:
        return explicit
    # Derive from PUBLIC_BASE_URL (which points at the frontend)
    base = PUBLIC_BASE_URL.rstrip("/")
    if "onrender.com" in base:
        return "https://zografia-backend.onrender.com"
    return base.replace("zografia-ai", "zografia-backend")


# --- Strict preservation prompt (from product spec PDF) ----------------------
# Goal: subtle motion ONLY. AKOOL must NOT redesign, recolor or restyle the
# child's drawing. Composed from a fixed preservation core + an optional
# what_i_see hint from OpenAI Vision so motion suggestions match the picture.
NEGATIVE_PROMPT = (
    "no style change, no recolor, no new objects, no text overlay, no extra "
    "effects, no character redesign, no warping, no morphing, no AI artifacts, "
    "no extra characters, no background change, no camera zoom, no crop"
)


def _strict_preservation_prompt(what_i_see: str = "", custom_motion: str = "") -> str:
    base = (
        "Animate this real child drawing while fully preserving the original image, "
        "original colors, original crayon style, original lines, and original "
        "composition. Do not redesign, repaint, recolor, or restyle anything. "
        "Keep the exact same characters, the same background, and the same overall drawing."
    )
    motion = (
        " Only add gentle motion: characters move slightly as if happily standing or "
        "playing together, arms move a little, bodies sway softly. Any sun or sky "
        "elements have a subtle happy motion. Grass and small background details can "
        "move slightly as if in a soft breeze."
    )
    tail = (
        " Keep everything sweet, child-safe, simple, and natural. No new objects, "
        "no style change, no extra effects, no text, 5 seconds."
    )
    parts = [base]
    if what_i_see:
        clean_see = what_i_see.replace("\n", " ").strip()[:220]
        parts.append(f" The drawing shows: {clean_see}.")
    parts.append(motion)
    if custom_motion:
        clean_motion = custom_motion.replace("\n", " ").strip()[:240]
        # Wrap the user's request so AKOOL still respects preservation: the parent's
        # hint becomes a motion *suggestion*, never a redesign instruction.
        parts.append(
            f" Parent's gentle motion hint (respect it as motion only, never as "
            f"a redesign): {clean_motion}."
        )
    parts.append(tail)
    return "".join(parts)


def _akool_create_image_to_video(image_url: str, what_i_see: str = "",
                                 custom_motion: str = "") -> dict:
    """Kick off an Image-to-Video task via the BATCH endpoint with count=1.

    The batch endpoint is the only one that accepts `model_name`, which lets
    us pick the cheap Akool Basic model (4 credits/720p) instead of the
    default premium model (100 credits) the single endpoint locks us into.
    """
    if not AKOOL_API_KEY:
        raise HTTPException(status_code=503, detail="AKOOL_API_KEY not set")
    payload = {
        "image_url": image_url,
        "prompt": _strict_preservation_prompt(what_i_see, custom_motion),
        "negative_prompt": NEGATIVE_PROMPT,
        "model_name": AKOOL_MODEL_NAME,
        "resolution": AKOOL_RESOLUTION,
        "video_length": AKOOL_VIDEO_LENGTH,
        "count": 1,
    }
    headers = {"x-api-key": AKOOL_API_KEY, "Content-Type": "application/json"}
    try:
        with httpx.Client(timeout=30.0) as client:
            r = client.post(f"{AKOOL_I2V_BASE}/createBySourcePrompt/batch",
                            json=payload, headers=headers)
            return r.json() if r.content else {}
    except httpx.HTTPError as e:
        print(f"[akool i2v start] http error: {e!r}")
        raise HTTPException(status_code=502, detail=f"AKOOL start error: {type(e).__name__}")


def _akool_status(video_id: str) -> dict:
    """Poll a single video by its _id. Returns AKOOL's raw JSON."""
    if not AKOOL_API_KEY:
        raise HTTPException(status_code=503, detail="AKOOL_API_KEY not set")
    headers = {"x-api-key": AKOOL_API_KEY, "Content-Type": "application/json"}
    try:
        with httpx.Client(timeout=20.0) as client:
            r = client.post(f"{AKOOL_I2V_BASE}/resultsByIds",
                            json={"_ids": video_id}, headers=headers)
            return r.json() if r.content else {}
    except httpx.HTTPError as e:
        print(f"[akool i2v status] http error: {e!r}")
        raise HTTPException(status_code=502, detail=f"AKOOL status error: {type(e).__name__}")


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "service": "zografia-backend",
        "openai_configured": bool(os.environ.get("OPENAI_API_KEY", "").strip()),
        "stripe_configured": bool(STRIPE_SECRET_KEY),
        "prices_configured": all([PRICE_BASIC_MONTHLY, PRICE_FULL_MONTHLY, PRICE_FULL_YEARLY]),
        "akool_configured":  bool(AKOOL_API_KEY),
    }


@app.get("/api/download-video")
def download_video_proxy(url: str):
    """Proxy AKOOL CDN videos back to the client with
    Content-Disposition: attachment so the browser triggers a real Save
    dialog with the right .mp4 filename. Bypasses Windows Media Player
    quirks where the same file fetched directly fails to open.

    Whitelisted to AKOOL CDN hosts only so we can't be used as an open
    redirect / bandwidth abuse target."""
    if not isinstance(url, str) or not url.startswith("https://"):
        raise HTTPException(status_code=400, detail="https url required")
    allowed_hosts = ("cloudfront.net", "akool.com", "akoolai.com")
    if not any(h in url for h in allowed_hosts):
        raise HTTPException(status_code=400, detail="untrusted host")
    try:
        with httpx.Client(timeout=60.0, follow_redirects=True) as client:
            r = client.get(url)
            r.raise_for_status()
            return StreamingResponse(
                io.BytesIO(r.content),
                media_type="video/mp4",
                headers={
                    "Content-Disposition": 'attachment; filename="zografia-zoi.mp4"',
                    "Cache-Control": "no-store",
                    "Content-Length": str(len(r.content)),
                },
            )
    except httpx.HTTPError as e:
        print(f"[download-video] http err: {e!r}")
        raise HTTPException(status_code=502, detail=f"fetch failed: {type(e).__name__}")
    except Exception as e:  # noqa: BLE001
        print(f"[download-video] err: {e!r}")
        raise HTTPException(status_code=502, detail="fetch failed")


@app.get("/api/animate-status")
def animate_status(task_id: str):
    """Poll AKOOL Image-to-Video for a task by its `_id`.
       Returns {status, video_url?, raw_code}."""
    if not task_id:
        raise HTTPException(status_code=400, detail="task_id required")
    data = _akool_status(task_id)
    # v4 shape: data.data.result[0] = {status, video_url, ...}
    item = {}
    if isinstance(data, dict):
        body = data.get("data") or {}
        if isinstance(body, dict):
            arr = body.get("result") or []
            if isinstance(arr, list) and arr:
                item = arr[0] if isinstance(arr[0], dict) else {}
    raw_vstat = item.get("status", 0)
    try: vstat = int(raw_vstat or 0)
    except (TypeError, ValueError): vstat = 0
    video_url = item.get("video_url") or ""
    state_str = {1: "queued", 2: "processing", 3: "success", 4: "failed"}.get(vstat, "unknown")
    return {
        "status": state_str,
        "video_url": video_url if state_str == "success" else "",
        "raw_code": data.get("code") if isinstance(data, dict) else None,
    }


@app.get("/api/usage")
def usage(x_device_id: Optional[str] = Header(None, alias="X-Device-Id")):
    dev = _require_device(x_device_id)
    dev = _refresh_billing_period(dev)
    return _plan_status_dict(dev)


# --- TTS --------------------------------------------------------------------
ALLOWED_VOICES = {"alloy", "echo", "fable", "onyx", "nova", "shimmer"}
DEFAULT_VOICE = "shimmer"
MAX_TTS_CHARS = 1200


class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = None
    speed: Optional[float] = None


def _generate_tts_mp3(text: str, voice: str = DEFAULT_VOICE, speed: float = 0.95) -> bytes:
    """Synchronously generate MP3 audio bytes from OpenAI TTS. Raises HTTPException on failure."""
    text = (text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")
    if len(text) > MAX_TTS_CHARS:
        text = text[:MAX_TTS_CHARS]
    voice = (voice or DEFAULT_VOICE).strip().lower()
    if voice not in ALLOWED_VOICES:
        voice = DEFAULT_VOICE
    try: speed = float(speed)
    except (TypeError, ValueError): speed = 0.95
    speed = max(0.5, min(1.5, speed))
    try:
        response = openai_client().audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text,
            response_format="mp3",
            speed=speed,
        )
        return response.read()
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        print(f"[tts] error: {exc!r}")
        raise HTTPException(status_code=502, detail=f"TTS error: {type(exc).__name__}")


@app.post("/api/tts")
def tts(req: TTSRequest, x_device_id: Optional[str] = Header(None, alias="X-Device-Id")):
    if x_device_id:
        try: _require_device(x_device_id)
        except HTTPException: pass
    audio_bytes = _generate_tts_mp3(
        req.text or "",
        req.voice or DEFAULT_VOICE,
        req.speed if req.speed is not None else 0.95,
    )
    return StreamingResponse(
        io.BytesIO(audio_bytes),
        media_type="audio/mpeg",
        headers={"Cache-Control": "no-store"},
    )


# --- Animate drawing (Vision + Story) --------------------------------------
MAX_IMAGE_BYTES = 6 * 1024 * 1024   # 6MB sent from frontend (already downscaled)


class AnimateRequest(BaseModel):
    image: str                              # data URL "data:image/...;base64,..."
    child_name: Optional[str] = None
    title: Optional[str] = None
    mood: Optional[str] = None              # "χαρούμενο" | "ήρεμο" | "παιχνιδιάρικο"
    custom_motion: Optional[str] = None     # optional Greek hint, e.g. "ο ήλιος να χαμογελάει"
    lang: Optional[str] = "el"              # "el" (default) or "en" — controls story + voice


class OwnerUnlockRequest(BaseModel):
    email: str


@app.post("/api/owner-unlock")
def owner_unlock(req: OwnerUnlockRequest,
                 x_device_id: Optional[str] = Header(None, alias="X-Device-Id")):
    """Marks the calling device as 'owner' (unlimited usage) if the supplied
    email is on the OWNER_EMAILS allow-list. For event/presentation use by
    the platform owner only — not exposed in marketing UI."""
    dev = _require_device(x_device_id)
    email = (req.email or "").strip().lower()
    if not email or "@" not in email:
        raise HTTPException(status_code=400, detail="email required")
    if email not in OWNER_EMAILS:
        # Don't reveal whether the email is in the list or not — generic 403
        raise HTTPException(status_code=403, detail="not_allowed")
    with db() as con:
        con.execute(
            """UPDATE devices SET plan='owner', plan_status='active',
                   uses=0, period_start=?, current_period_end=0, updated_at=?
               WHERE id=?""",
            (_now(), _now(), dev["id"]),
        )
        row = con.execute("SELECT * FROM devices WHERE id=?", (dev["id"],)).fetchone()
    return _plan_status_dict(dict(row))


SYSTEM_PROMPT_ANIMATE_EL = (
    "Είσαι μια χαρούμενη και ασφαλής Ελληνίδα αφηγήτρια παιδικών παραμυθιών.\n"
    "Σου δείχνουν μια ζωγραφιά παιδιού. Πρέπει να τη «ζωντανέψεις» — να γράψεις\n"
    "ένα μικρό μαγικό κείμενο (3–5 προτάσεις) και 2–3 σύντομα «λόγια» που θα πει\n"
    "η ζωγραφιά στο παιδί.\n\n"
    "Κανόνες:\n"
    "- ΟΛΑ στα Ελληνικά, με απλά λόγια, ζεστά, σαν φίλη.\n"
    "- Ασφαλές, χαρούμενο, παιχνιδιάρικο. Καμία τρομαχτική, βίαιη ή ακατάλληλη αναφορά.\n"
    "- Αν υπάρχει όνομα παιδιού, χρησιμοποίησέ το άπαξ φυσικά.\n"
    "- Επαίνεσε ευγενικά τη δουλειά («μου αρέσει που έβαλες…»). Όχι κολακείες ψεύτικες.\n"
    "- Περίγραψε σύντομα τι βλέπεις στη ζωγραφιά (απλά, αναγνωρίσιμα στοιχεία).\n"
    "- Αν δεν είσαι σίγουρη τι είναι κάτι, πες το ευγενικά («μου φαίνεται σαν…»).\n"
    "- Απάντησε ΜΟΝΟ με έγκυρο JSON ακριβώς αυτής της δομής:"
)
JSON_SHAPE_ANIMATE_EL = (
    '{\n'
    '  "title": "Σύντομος τίτλος (έως 6 λέξεις)",\n'
    '  "what_i_see": "Σύντομη ανθρώπινη περιγραφή του τι βλέπω.",\n'
    '  "story": "3–5 προτάσεις. Μαγική, ζεστή, αφηγηματική.",\n'
    '  "lines": [\n'
    '    {"speaker": "πχ ο ήλιος", "text": "1 σύντομη πρόταση που λέει στο παιδί"},\n'
    '    {"speaker": "πχ ο σκύλος", "text": "Ακόμα μια σύντομη πρόταση"}\n'
    '  ],\n'
    '  "mood": "χαρούμενο | ήρεμο | παιχνιδιάρικο | μαγικό",\n'
    '  "follow_up": "Μια ζεστή πρόταση/ερώτηση για το παιδί (π.χ. «θες να μου πεις πού πάει ο ήλιος;»)"\n'
    '}'
)

SYSTEM_PROMPT_ANIMATE_EN = (
    "You are a cheerful, gentle storyteller for young children.\n"
    "A child's drawing has been shared with you. Your job is to 'bring it to\n"
    "life' — write a short magical paragraph (3–5 sentences) and 2–3 short\n"
    "lines that the drawing would say to the child.\n\n"
    "Rules:\n"
    "- EVERYTHING in English, simple words, warm, like a friend.\n"
    "- Safe, joyful, playful. Never anything scary, violent or inappropriate.\n"
    "- If a child name is provided, use it once, naturally.\n"
    "- Praise the work kindly (\"I love how you drew…\"). No empty flattery.\n"
    "- Briefly describe what you see in the drawing (simple, recognizable parts).\n"
    "- If unsure what something is, say so gently (\"it looks like a…\").\n"
    "- Respond ONLY with valid JSON of exactly this shape:"
)
JSON_SHAPE_ANIMATE_EN = (
    '{\n'
    '  "title": "Short title (up to 6 words)",\n'
    '  "what_i_see": "Short human description of what I see.",\n'
    '  "story": "3–5 sentences. Magical, warm, narrative.",\n'
    '  "lines": [\n'
    '    {"speaker": "e.g. the sun", "text": "1 short sentence the sun says to the child"},\n'
    '    {"speaker": "e.g. the dog", "text": "Another short sentence"}\n'
    '  ],\n'
    '  "mood": "happy | calm | playful | magical",\n'
    '  "follow_up": "A warm question for the child (e.g. \\"Where do you think the sun is going next?\\")"\n'
    '}'
)


def _system_and_shape(lang: str):
    return (SYSTEM_PROMPT_ANIMATE_EN, JSON_SHAPE_ANIMATE_EN) if (lang or "").lower() == "en" \
           else (SYSTEM_PROMPT_ANIMATE_EL, JSON_SHAPE_ANIMATE_EL)


def _extract_json(text: str) -> dict:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    try: return json.loads(text)
    except Exception:
        m = re.search(r"\{[\s\S]*\}", text)
        if m: return json.loads(m.group(0))
        raise


def _validate_animation(obj: dict, lang: str = "el") -> dict:
    if not isinstance(obj, dict): raise ValueError("not an object")
    story = (obj.get("story") or "").strip()
    if not story: raise ValueError("missing story")
    lines = obj.get("lines") or []
    if not isinstance(lines, list): lines = []
    # Language-appropriate default speakers / fallback lines
    if (lang or "el").lower() == "en":
        default_speaker = "the drawing"
        default_line    = "Hi! You drew me beautifully!"
        default_title   = "My drawing"
        default_mood    = "happy"
    else:
        default_speaker = "η ζωγραφιά"
        default_line    = "Γεια σου! Με ζωγράφισες πολύ ωραία!"
        default_title   = "Η ζωγραφιά μου"
        default_mood    = "χαρούμενο"
    cleaned_lines = []
    for ln in lines[:5]:
        if not isinstance(ln, dict): continue
        sp = str(ln.get("speaker") or "").strip()
        tx = str(ln.get("text") or "").strip()
        if not tx: continue
        cleaned_lines.append({"speaker": sp[:60] or default_speaker, "text": tx[:280]})
    return {
        "title":      (str(obj.get("title") or "").strip() or default_title)[:80],
        "what_i_see": str(obj.get("what_i_see") or "").strip()[:400],
        "story":      story[:1200],
        "lines":      cleaned_lines or [{"speaker": default_speaker, "text": default_line}],
        "mood":       (str(obj.get("mood") or default_mood).strip().lower())[:30],
        "follow_up":  str(obj.get("follow_up") or "").strip()[:280],
    }


def _validate_image_data_url(image: str) -> str:
    """Reject obviously-bad / over-large payloads early."""
    if not image or not isinstance(image, str):
        raise HTTPException(status_code=400, detail="image is required")
    if not image.startswith("data:image/"):
        raise HTTPException(status_code=400, detail="image must be a data URL")
    # rough size cap (base64 ~= 1.37x raw bytes)
    if len(image) > MAX_IMAGE_BYTES * 4 // 3 + 200:
        raise HTTPException(status_code=413, detail="image too large (max ~6MB)")
    # quick sanity: must contain base64,
    if ";base64," not in image:
        raise HTTPException(status_code=400, detail="image must be base64-encoded")
    return image


def _speakable_full_text(data: dict) -> str:
    """Combine story + lines into one nice script for TTS playback."""
    parts = [data.get("story", "").strip()]
    for ln in data.get("lines") or []:
        sp = (ln.get("speaker") or "").strip()
        tx = (ln.get("text") or "").strip()
        if not tx: continue
        if sp:
            parts.append(f"{sp.capitalize()}: {tx}")
        else:
            parts.append(tx)
    fu = (data.get("follow_up") or "").strip()
    if fu: parts.append(fu)
    return "\n\n".join(p for p in parts if p)


@app.post("/api/animate-drawing")
def animate_drawing(req: AnimateRequest,
                    x_device_id: Optional[str] = Header(None, alias="X-Device-Id")):
    _require_device(x_device_id)
    image = _validate_image_data_url(req.image)

    # Atomic quota check + consume BEFORE the (expensive) Vision call
    dev_after = _consume(x_device_id)

    child = (req.child_name or "").strip()[:40]
    title = (req.title or "").strip()[:80]
    mood  = (req.mood or "").strip()[:20]
    lang  = (req.lang or "el").strip().lower()
    if lang not in ("el", "en"): lang = "el"
    sys_prompt, json_shape = _system_and_shape(lang)

    extra_bits = []
    if lang == "en":
        if child: extra_bits.append(f"Child's name: {child}.")
        if title: extra_bits.append(f"Drawing title (if it helps): {title}.")
        if mood:  extra_bits.append(f"Desired mood: {mood}.")
        opener  = "Look at this child's drawing and bring it to life."
        reminder = "\n\nReply ONLY with JSON of this shape:\n"
    else:
        if child: extra_bits.append(f"Όνομα παιδιού: {child}.")
        if title: extra_bits.append(f"Τίτλος ζωγραφιάς (αν σε βοηθάει): {title}.")
        if mood:  extra_bits.append(f"Επιθυμητή διάθεση: {mood}.")
        opener  = "Δες αυτή τη ζωγραφιά παιδιού και ζωντάνεψέ τη."
        reminder = "\n\nΑπάντησε ΜΟΝΟ με JSON αυτής της δομής:\n"
    extra_block = ("\n" + "\n".join(extra_bits)) if extra_bits else ""

    user_prompt = opener + extra_block + reminder + json_shape

    try:
        completion = openai_client().chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": image, "detail": "low"}},
                ]},
            ],
            response_format={"type": "json_object"},
            temperature=0.85,
            max_tokens=900,
        )
        raw = completion.choices[0].message.content or ""
        data = _extract_json(raw)
        out = _validate_animation(data, lang)
        out["speakable"] = _speakable_full_text(out)
        out["usage"]     = _plan_status_dict(dev_after)
        out["lang"]      = lang
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        print(f"[animate-drawing] error: {exc!r}")
        _refund(x_device_id)
        raise HTTPException(status_code=502, detail=f"OpenAI: {type(exc).__name__}: {str(exc)[:280]}")

    # --- generate the spoken audio ONCE here, save it to disk, return its URL.
    # The frontend will play that file directly (so we don't pay for OpenAI TTS
    # twice). The same URL is also handed to AKOOL.
    out["audio_url"]     = ""
    out["video_task_id"] = ""
    out["video_status"]  = "none"

    backend_host = _backend_origin()

    audio_url = ""
    try:
        mp3_bytes = _generate_tts_mp3(out["speakable"])
        audio_name = _save_bytes(mp3_bytes, "mp3")
        audio_url = f"{backend_host}/uploads/{audio_name}"
        out["audio_url"] = audio_url
    except HTTPException as he:
        print(f"[animate-drawing TTS] skipped: {he.detail}")
    except Exception as exc:
        print(f"[animate-drawing TTS] error: {exc!r}")

    # --- kick off the AKOOL Image-to-Video task so the frontend can start
    # polling immediately. We send the original drawing + a strict preservation
    # prompt — AKOOL must NOT redesign or recolor the child's art, only add
    # gentle motion. Audio is played in the browser, NOT mixed into the video.
    if AKOOL_API_KEY:
        try:
            img_name = _save_image_to_disk(image)
            image_url = f"{backend_host}/uploads/{img_name}"
            akool_resp = _akool_create_image_to_video(
                image_url=image_url,
                what_i_see=out.get("what_i_see") or "",
                custom_motion=(req.custom_motion or "").strip(),
            )
            video_id = ""
            if isinstance(akool_resp, dict):
                body = akool_resp.get("data") or {}
                # Batch endpoint shape: data.successList[0]._id
                slist = body.get("successList") if isinstance(body, dict) else None
                if isinstance(slist, list) and slist and isinstance(slist[0], dict):
                    video_id = slist[0].get("_id") or ""
                # Fallback (single-create shape, if we ever switch back)
                if not video_id and isinstance(body, dict):
                    video_id = body.get("_id") or ""
                if (akool_resp.get("code") not in (1000, None)) and not video_id:
                    print(f"[animate-drawing AKOOL] non-OK: {akool_resp}")
            if video_id:
                out["video_task_id"] = video_id
                out["video_status"]  = "queued"
        except HTTPException as he:
            print(f"[animate-drawing AKOOL] skipped: {he.detail}")
        except Exception as exc:
            print(f"[animate-drawing AKOOL] error: {exc!r}")
    return out


# --- Stripe checkout --------------------------------------------------------
class CheckoutRequest(BaseModel):
    plan: str  # "basic_monthly" | "full_monthly" | "full_yearly"


PRICE_LOOKUP = {
    "basic_monthly": PRICE_BASIC_MONTHLY,
    "full_monthly":  PRICE_FULL_MONTHLY,
    "full_yearly":   PRICE_FULL_YEARLY,
}


@app.post("/api/checkout")
def checkout(req: CheckoutRequest,
             x_device_id: Optional[str] = Header(None, alias="X-Device-Id")):
    dev = _require_device(x_device_id)
    price_id = PRICE_LOOKUP.get(req.plan, "")
    if not req.plan in PRICE_LOOKUP:
        raise HTTPException(status_code=400, detail="Unknown plan")
    if not price_id:
        raise HTTPException(status_code=503, detail=f"{req.plan} not configured")

    try:
        s = stripe_module()
        session_kwargs = dict(
            mode="subscription",
            line_items=[{"price": price_id, "quantity": 1}],
            success_url=f"{PUBLIC_BASE_URL}/?checkout=success&plan={req.plan}",
            cancel_url=f"{PUBLIC_BASE_URL}/?checkout=cancel",
            client_reference_id=dev["id"],
            metadata={"device_id": dev["id"], "plan": req.plan},
            allow_promotion_codes=True,
        )
        if dev.get("stripe_customer_id"):
            session_kwargs["customer"] = dev["stripe_customer_id"]
        session = s.checkout.Session.create(**session_kwargs)
        return {"url": session.url}
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        print(f"[checkout] error: {exc!r}")
        raise HTTPException(status_code=502, detail=f"Stripe: {type(exc).__name__}: {str(exc)[:240]}")


# --- Stripe webhook ---------------------------------------------------------
@app.post("/api/stripe-webhook")
async def stripe_webhook(request: Request):
    if not STRIPE_WEBHOOK_SECRET:
        raise HTTPException(status_code=503, detail="webhook secret not configured")
    s = stripe_module()
    payload = await request.body()
    sig = request.headers.get("stripe-signature", "")
    try:
        event = s.Webhook.construct_event(payload, sig, STRIPE_WEBHOOK_SECRET)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"bad signature: {exc}")

    etype = event["type"]
    obj   = event["data"]["object"]

    def _activate(device_id: str, sub):
        plan_name = "basic"
        try:
            items = sub.get("items", {}).get("data", []) if isinstance(sub, dict) else []
            if items:
                price_id = items[0].get("price", {}).get("id", "")
                plan_name = PLAN_FROM_PRICE.get(price_id, "basic")
        except Exception: pass
        cpe = int(sub.get("current_period_end") or 0) if isinstance(sub, dict) else 0
        with db() as con:
            con.execute(
                """UPDATE devices SET
                       plan=?, plan_status='active',
                       current_period_end=?, uses=0, period_start=?,
                       stripe_customer_id=?, stripe_subscription_id=?,
                       updated_at=?
                   WHERE id=?""",
                (plan_name, cpe, _now(),
                 (sub.get("customer") if isinstance(sub, dict) else "") or "",
                 (sub.get("id") if isinstance(sub, dict) else "") or "",
                 _now(), device_id),
            )

    def _deactivate_by_sub(sub_id: str):
        if not sub_id: return
        with db() as con:
            con.execute(
                """UPDATE devices SET plan='trial', plan_status='canceled', updated_at=?
                   WHERE stripe_subscription_id=?""",
                (_now(), sub_id),
            )

    try:
        if etype == "checkout.session.completed":
            device_id = (obj.get("metadata") or {}).get("device_id") or obj.get("client_reference_id") or ""
            sub_id = obj.get("subscription") or ""
            if device_id and sub_id:
                sub = s.Subscription.retrieve(sub_id)
                _activate(device_id, sub if isinstance(sub, dict) else dict(sub))
        elif etype in ("customer.subscription.created", "customer.subscription.updated"):
            sub = obj
            sub_id = sub.get("id") or ""
            device_id = (sub.get("metadata") or {}).get("device_id") or ""
            if not device_id and sub_id:
                with db() as con:
                    row = con.execute(
                        "SELECT id FROM devices WHERE stripe_subscription_id=?",
                        (sub_id,),
                    ).fetchone()
                    if row: device_id = row["id"]
            if device_id:
                status = sub.get("status", "")
                if status in ("active", "trialing"):
                    _activate(device_id, sub)
                else:
                    _deactivate_by_sub(sub_id)
        elif etype == "customer.subscription.deleted":
            _deactivate_by_sub(obj.get("id") or "")
    except Exception as exc:
        print(f"[webhook {etype}] error: {exc!r}")

    return {"received": True}
