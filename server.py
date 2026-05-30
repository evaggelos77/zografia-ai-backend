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

TRIAL_LIMIT = 2
BASIC_LIMIT = 5   # per Stripe billing period
FULL_LIMIT  = 15  # per Stripe billing period (full monthly)
YEARLY_LIMIT = 180  # full yearly cap (~15/month) — Phase 2 will offer top-ups

AKOOL_API_KEY = os.environ.get("AKOOL_API_KEY", "").strip()
AKOOL_BASE    = "https://openapi.akool.com/api/open/v3"
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
    if plan == "yearly": return YEARLY_LIMIT
    if plan == "full":   return FULL_LIMIT
    if plan == "basic":  return BASIC_LIMIT
    return TRIAL_LIMIT


def _plan_status_dict(dev: dict) -> dict:
    plan = dev.get("plan") or "trial"
    status = dev.get("plan_status") or "active"
    is_active = status == "active"
    is_paid = plan in ("basic", "full", "yearly") and is_active
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
# --- AKOOL Image-to-Video ---------------------------------------------------
def _save_image_to_disk(data_url: str) -> str:
    """Decode a data URL, save under UPLOAD_DIR, return the public name (without ext)."""
    if not data_url or ";base64," not in data_url:
        raise HTTPException(status_code=400, detail="image data URL required")
    b64 = data_url.split(";base64,", 1)[1]
    try:
        raw = base64.b64decode(b64)
    except Exception:
        raise HTTPException(status_code=400, detail="image not base64-decodable")
    if len(raw) > 6 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="image too large")
    name = secrets.token_urlsafe(18)
    with open(os.path.join(UPLOAD_DIR, name + ".jpg"), "wb") as f:
        f.write(raw)
    return name


@app.get("/uploads/{name}.jpg")
def get_upload(name: str):
    safe = re.sub(r"[^A-Za-z0-9_-]", "", name)[:64]
    if not safe:
        raise HTTPException(status_code=404, detail="not found")
    path = os.path.join(UPLOAD_DIR, safe + ".jpg")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=86400"})


# AKOOL Image-to-Video model IDs (v3 OpenAPI). 401 = standard / cheaper plan.
AKOOL_MODEL_ID = int(os.environ.get("AKOOL_MODEL_ID", "401"))


def _akool_start(image_public_url: str, prompt: str = "") -> dict:
    if not AKOOL_API_KEY:
        raise HTTPException(status_code=503, detail="AKOOL_API_KEY not set")
    payload = {
        "url": image_public_url,
        "model_id": AKOOL_MODEL_ID,
        "prompt": (prompt or "Gentle, subtle motion. Do not change colors, lines or style. No new objects, no zoom, no crop.")[:500],
        "duration": 5,
    }
    headers = {"Authorization": f"Bearer {AKOOL_API_KEY}", "Content-Type": "application/json"}
    try:
        with httpx.Client(timeout=30.0) as client:
            r = client.post(f"{AKOOL_BASE}/content/video/createbyimage", json=payload, headers=headers)
            r.raise_for_status()
            return r.json() or {}
    except httpx.HTTPError as e:
        print(f"[akool start] http error: {e!r}")
        raise HTTPException(status_code=502, detail=f"AKOOL start error: {type(e).__name__}")


def _akool_status(task_id: str) -> dict:
    if not AKOOL_API_KEY:
        raise HTTPException(status_code=503, detail="AKOOL_API_KEY not set")
    headers = {"Authorization": f"Bearer {AKOOL_API_KEY}"}
    try:
        with httpx.Client(timeout=20.0) as client:
            r = client.get(f"{AKOOL_BASE}/content/video/infobymodelid",
                          params={"task_id": task_id}, headers=headers)
            r.raise_for_status()
            return r.json() or {}
    except httpx.HTTPError as e:
        print(f"[akool status] http error: {e!r}")
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


@app.get("/api/animate-status")
def animate_status(task_id: str):
    """Poll AKOOL for a video task. Returns {status, video_url?, raw}."""
    if not task_id:
        raise HTTPException(status_code=400, detail="task_id required")
    data = _akool_status(task_id)
    body = (data.get("data") or {}) if isinstance(data, dict) else {}
    # AKOOL video_status: 1=queueing 2=processing 3=success 4=failed
    vstat = body.get("video_status") or body.get("status") or 0
    video_url = body.get("video") or body.get("video_url") or ""
    state_str = {1: "queued", 2: "processing", 3: "success", 4: "failed"}.get(int(vstat) if vstat else 0, "unknown")
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


@app.post("/api/tts")
def tts(req: TTSRequest, x_device_id: Optional[str] = Header(None, alias="X-Device-Id")):
    if x_device_id:
        try: _require_device(x_device_id)
        except HTTPException: pass

    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")
    if len(text) > MAX_TTS_CHARS:
        text = text[:MAX_TTS_CHARS]

    voice = (req.voice or DEFAULT_VOICE).strip().lower()
    if voice not in ALLOWED_VOICES:
        voice = DEFAULT_VOICE

    speed = req.speed if req.speed is not None else 0.95
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
        audio_bytes = response.read()
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        print(f"[tts] error: {exc!r}")
        raise HTTPException(status_code=502, detail=f"TTS error: {type(exc).__name__}")

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


SYSTEM_PROMPT_ANIMATE = (
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
JSON_SHAPE_ANIMATE = (
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


def _extract_json(text: str) -> dict:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    try: return json.loads(text)
    except Exception:
        m = re.search(r"\{[\s\S]*\}", text)
        if m: return json.loads(m.group(0))
        raise


def _validate_animation(obj: dict) -> dict:
    if not isinstance(obj, dict): raise ValueError("not an object")
    story = (obj.get("story") or "").strip()
    if not story: raise ValueError("missing story")
    lines = obj.get("lines") or []
    if not isinstance(lines, list): lines = []
    cleaned_lines = []
    for ln in lines[:5]:
        if not isinstance(ln, dict): continue
        sp = str(ln.get("speaker") or "").strip()
        tx = str(ln.get("text") or "").strip()
        if not tx: continue
        cleaned_lines.append({"speaker": sp[:60] or "η ζωγραφιά", "text": tx[:280]})
    return {
        "title":      (str(obj.get("title") or "").strip() or "Η ζωγραφιά μου")[:80],
        "what_i_see": str(obj.get("what_i_see") or "").strip()[:400],
        "story":      story[:1200],
        "lines":      cleaned_lines or [{"speaker": "η ζωγραφιά", "text": "Γεια σου! Με ζωγράφισες πολύ ωραία!"}],
        "mood":       (str(obj.get("mood") or "χαρούμενο").strip().lower())[:30],
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

    extra_bits = []
    if child:  extra_bits.append(f"Όνομα παιδιού: {child}.")
    if title:  extra_bits.append(f"Τίτλος ζωγραφιάς (αν σε βοηθάει): {title}.")
    if mood:   extra_bits.append(f"Επιθυμητή διάθεση: {mood}.")
    extra_block = ("\n" + "\n".join(extra_bits)) if extra_bits else ""

    user_prompt = (
        "Δες αυτή τη ζωγραφιά παιδιού και ζωντάνεψέ τη.\n"
        + extra_block
        + "\n\nΑπάντησε ΜΟΝΟ με JSON αυτής της δομής:\n"
        + JSON_SHAPE_ANIMATE
    )

    try:
        completion = openai_client().chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_ANIMATE},
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
        out = _validate_animation(data)
        out["speakable"] = _speakable_full_text(out)
        out["usage"]     = _plan_status_dict(dev_after)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        print(f"[animate-drawing] error: {exc!r}")
        _refund(x_device_id)
        raise HTTPException(status_code=502, detail=f"OpenAI: {type(exc).__name__}: {str(exc)[:280]}")

    # --- kick off the AKOOL video in the SAME call so the frontend can start
    # polling immediately. If AKOOL fails or isn't configured, we still return
    # the story (talking + sparkles still work).
    out["video_task_id"] = ""
    out["video_status"]  = "none"
    if AKOOL_API_KEY:
        try:
            img_name = _save_image_to_disk(image)
            public_url = f"{PUBLIC_BASE_URL.replace('zografia-ai', 'zografia-backend')}/uploads/{img_name}.jpg"
            # PUBLIC_BASE_URL points to the frontend; build the backend URL by
            # swapping the host. (Fallback if user kept default config.)
            backend_host = os.environ.get("BACKEND_PUBLIC_URL", "").rstrip("/")
            if backend_host:
                public_url = f"{backend_host}/uploads/{img_name}.jpg"
            elif "onrender.com" in PUBLIC_BASE_URL:
                public_url = f"https://zografia-backend.onrender.com/uploads/{img_name}.jpg"
            akool_resp = _akool_start(public_url, prompt=(out.get("what_i_see") or ""))
            task_id = ""
            if isinstance(akool_resp, dict):
                body = akool_resp.get("data") or {}
                task_id = body.get("task_id") or body.get("_id") or ""
            if task_id:
                out["video_task_id"] = task_id
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
