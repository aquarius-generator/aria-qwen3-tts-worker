"""RunPod Serverless handler for Qwen3-TTS (Apache-2.0) voice cloning.

ONE CONTRACT ACROSS BOTH VOICE WORKERS. This worker and the Chatterbox one take the same input and return
the same shape, so ARIA's lab calls either through one client and the blind comparison is a comparison of
voices rather than of two different payload formats.

THE ONE ASYMMETRY, AND IT IS THE MODEL'S, NOT A DESIGN CHOICE. Qwen's cloning call wants the TRANSCRIPT of
the reference clip as well as the clip itself: `ref_audio` plus `ref_text`. Chatterbox needs only the clip.
So `reference_text` is part of this contract, and a missing one is a hard error rather than an empty string
quietly passed through, because cloning against a wrong transcript degrades the voice in a way that is easy
to mistake for the model simply being worse.

THE REFERENCE CLIP IS AN INPUT, NOT A SETTING. There is no trained voice here; the clip is sent with every
request. A missing clip is a hard error, never a fallback to a default voice, because ten valid outputs in
the wrong voice look exactly like a working comparison.

WEIGHTS LOAD AT IMPORT, DURING WORKER INIT. The execution timeout bounds how long one REQUEST may run; it
is not a budget for bringing a worker up.
"""
import base64
import io
import os
import time
import traceback

import runpod

MODEL_ID = os.environ.get("VOICE_MODEL_ID", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
MODEL_LICENSE = os.environ.get("VOICE_MODEL_LICENSE", "Apache-2.0")

_model = None
_cache_kind = "unknown"
_load_seconds = 0.0
_init_error = None


def _choose_cache():
    """Network volume when one is mounted, container disk otherwise. Returned so it can be reported."""
    vol = os.environ.get("VOICE_VOLUME_CACHE", "/runpod-volume/hf")
    mount = os.path.dirname(vol.rstrip("/")) or "/runpod-volume"
    if os.path.isdir(mount) and os.access(mount, os.W_OK):
        os.makedirs(vol, exist_ok=True)
        return vol, "network-volume"
    fb = os.environ.get("VOICE_FALLBACK_CACHE", "/app/hf-cache")
    os.makedirs(fb, exist_ok=True)
    return fb, "container-disk"


def _load():
    global _model, _cache_kind, _load_seconds
    if _model is not None:
        return _model

    cache, _cache_kind = _choose_cache()
    os.environ["HF_HOME"] = cache
    os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(cache, "hub")
    print(f"[worker] cache {cache} ({_cache_kind})", flush=True)

    import torch
    from qwen_tts import Qwen3TTSModel

    started = time.time()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # No attn_implementation override: the default avoids a flash-attn source build, which would put this
    # image back under the build ceiling for no gain on short utterances.
    _model = Qwen3TTSModel.from_pretrained(MODEL_ID, device_map=device, dtype=torch.bfloat16)
    _load_seconds = round(time.time() - started, 1)
    print(f"[worker] Qwen3-TTS ready on {device} in {_load_seconds}s", flush=True)
    return _model


def _write_reference(ref):
    if not ref or not ref.get("base64"):
        raise ValueError("A reference_audio clip is required; this worker never falls back to a default voice.")
    raw = base64.b64decode(ref["base64"])
    if not raw:
        raise ValueError("The reference_audio clip decoded to zero bytes.")
    ctype = str(ref.get("content_type") or "audio/wav").lower()
    ext = {"audio/wav": "wav", "audio/x-wav": "wav", "audio/mpeg": "mp3", "audio/mp3": "mp3",
           "audio/flac": "flac", "audio/ogg": "ogg", "audio/mp4": "m4a", "audio/m4a": "m4a"}.get(ctype, "wav")
    path = f"/tmp/reference.{ext}"
    with open(path, "wb") as fh:
        fh.write(raw)
    return path, len(raw)


def handler(job):
    started = time.time()
    try:
        payload = job.get("input") or {}
        text = (payload.get("text") or "").strip()
        if not text:
            return {"error": "Text is required."}

        ref_text = (payload.get("reference_text") or "").strip()
        if not ref_text:
            # Refused rather than defaulted. Cloning against a wrong or empty transcript degrades the voice
            # in a way that is easy to mistake for the model simply being worse than its rival.
            return {"error": "reference_text is required: this model clones from the clip AND its transcript."}

        ref_path, ref_bytes = _write_reference(payload.get("reference_audio"))
        cold = _model is None
        model = _load()

        import soundfile as sf
        import torch

        seed = payload.get("seed")
        if seed is not None:
            torch.manual_seed(int(seed))

        wavs, sr = model.generate_voice_clone(
            text=text,
            language=payload.get("language") or "English",
            ref_audio=ref_path,
            ref_text=ref_text,
        )
        arr = wavs[0]
        if hasattr(arr, "ndim") and arr.ndim > 1:
            arr = arr.squeeze()

        buf = io.BytesIO()
        sf.write(buf, arr, int(sr), format="WAV", subtype="PCM_16")
        raw = buf.getvalue()
        duration = round(len(arr) / float(sr), 2)

        return {
            "audio_base64": base64.b64encode(raw).decode("utf-8"),
            "mime_type": "audio/wav",
            "sample_rate": int(sr),
            "duration_seconds": duration,
            "characters": len(text),
            "reference_bytes": ref_bytes,
            "model": MODEL_ID,
            "license": MODEL_LICENSE,
            "weights_cache": _cache_kind,
            "cold_start": cold,
            "model_load_seconds": _load_seconds,
            "latency_seconds": round(time.time() - started, 2),
        }
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return {"error": f"{type(exc).__name__}: {exc}"}


# Warm during worker init, not inside the first job.
try:
    _load()
except Exception as _exc:  # noqa: BLE001
    _init_error = f"{type(_exc).__name__}: {_exc}"
    print(f"[worker] INIT FAILED: {_init_error}", flush=True)
    traceback.print_exc()

runpod.serverless.start({"handler": handler})
