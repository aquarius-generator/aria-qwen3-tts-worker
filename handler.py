"""RunPod Serverless handler for Qwen3-TTS (Apache-2.0): ONE endpoint, TWO modes.

  mode="voice_design"  text + a written DESCRIPTION of a voice -> speech in a voice nobody has ever had
  mode="voice_clone"   text + a reference clip and its transcript -> speech in that clip's voice

WHY ONE ENDPOINT. Both checkpoints declare `architectures: ["Qwen3TTSForConditionalGeneration"]`,
`model_type: qwen3_tts` and `transformers_version: 4.57.3`, so the pinned dependency set that loads one
loads the other with nothing changed. A second endpoint would buy nothing and idle its own GPU. The only
structural difference is that Base carries `speaker_encoder_config`, which is what lets it clone from
audio, and VoiceDesign has none because it takes a description instead. That difference IS the two modes.

CHECKPOINTS LOAD LAZILY, PER MODE, AND ARE KEPT. A batch in one mode never pays to load the other, and the
second request in a mode pays nothing at all. Both are pinned to immutable shas, and the sha that actually
answered is echoed on every response so the places that name it can never silently disagree.

VOICE DESIGN TAKES NO AUDIO, BY CONSTRUCTION. There is no reference clip in that path and no way to pass
one. That is the point of this worker: the voice it produces is designed from a sentence, not copied from
a person, so no recording of anybody is involved at any stage.
"""
import base64
import io
import os
import time
import traceback

import runpod

# Two checkpoints, two modes, both Apache-2.0. Pinned to immutable commits: a tag can be re-pointed
# upstream, and a comparison whose contestant changed between sittings proves nothing.
DESIGN_ID = os.environ.get("QWEN_DESIGN_MODEL_ID", "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
DESIGN_SHA = os.environ.get("QWEN_DESIGN_MODEL_SHA", "5ecdb67327fd37bb2e042aab12ff7391903235d3")
CLONE_ID = os.environ.get("QWEN_CLONE_MODEL_ID", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
CLONE_SHA = os.environ.get("QWEN_CLONE_MODEL_SHA", "fd4b254389122332181a7c3db7f27e918eec64e3")
MODEL_LICENSE = os.environ.get("VOICE_MODEL_LICENSE", "Apache-2.0")

MODES = {
    "voice_design": {"repo": DESIGN_ID, "sha": DESIGN_SHA},
    "voice_clone": {"repo": CLONE_ID, "sha": CLONE_SHA},
}

_models = {}          # mode -> loaded model
_load_seconds = {}    # mode -> seconds it took
_cache_kind = "unknown"
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


def _set_cache_env():
    global _cache_kind
    cache, _cache_kind = _choose_cache()
    # Set before the libraries are imported: they read these at import time.
    os.environ["HF_HOME"] = cache
    os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(cache, "hub")
    return cache


def _load(mode):
    """Load (once) the checkpoint for this mode. Both live on the same volume."""
    if mode in _models:
        return _models[mode]
    spec = MODES[mode]

    cache = _set_cache_env()
    print(f"[worker] loading {mode}: {spec['repo']} @ {spec['sha']} (cache {cache}, {_cache_kind})", flush=True)

    import torch
    from qwen_tts import Qwen3TTSModel

    started = time.time()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # No attn_implementation override: the default avoids a flash-attn source build, which would put this
    # image back under the build ceiling for no gain on short utterances.
    model = Qwen3TTSModel.from_pretrained(
        spec["repo"],
        revision=spec["sha"],
        device_map=device,
        dtype=torch.bfloat16,
    )
    _models[mode] = model
    _load_seconds[mode] = round(time.time() - started, 1)
    print(f"[worker] {mode} ready on {device} in {_load_seconds[mode]}s", flush=True)
    return model


def _write_reference(ref):
    if not ref or not ref.get("base64"):
        raise ValueError("voice_clone needs a reference_audio clip; this worker never falls back to a default voice.")
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
        mode = str(payload.get("mode") or "voice_clone").strip()
        if mode not in MODES:
            return {"error": "Unknown mode " + repr(mode) + ". Use voice_design or voice_clone."}

        text = (payload.get("text") or "").strip()
        if not text:
            return {"error": "Text is required."}

        language = payload.get("language") or "English"
        cold = mode not in _models
        model = _load(mode)

        import soundfile as sf
        import torch

        seed = payload.get("seed")
        if seed is not None:
            torch.manual_seed(int(seed))

        ref_bytes = 0
        if mode == "voice_design":
            instruct = (payload.get("instruct") or payload.get("voice_description") or "").strip()
            if not instruct:
                # Refused rather than defaulted. A design request with no description returns the model's
                # own default voice, which looks like a successful result and is not the voice anyone
                # asked for.
                return {"error": "voice_design needs an `instruct` description of the voice to design."}
            wavs, sr = model.generate_voice_design(text=text, instruct=instruct, language=language)
        else:
            ref_text = (payload.get("reference_text") or "").strip()
            if not ref_text:
                # Cloning against a missing transcript degrades the output in a way that is easy to
                # mistake for the model simply being worse than its rival.
                return {"error": "voice_clone needs `reference_text`: this model clones from the clip AND its transcript."}
            ref_path, ref_bytes = _write_reference(payload.get("reference_audio"))
            wavs, sr = model.generate_voice_clone(
                text=text, language=language, ref_audio=ref_path, ref_text=ref_text
            )

        arr = wavs[0]
        if hasattr(arr, "ndim") and arr.ndim > 1:
            arr = arr.squeeze()

        buf = io.BytesIO()
        sf.write(buf, arr, int(sr), format="WAV", subtype="PCM_16")
        raw = buf.getvalue()
        duration = round(len(arr) / float(sr), 2)

        # Bytes, never a URL. The caller stores these permanently and a signed link would expire long
        # before anyone listens to them.
        return {
            "audio_base64": base64.b64encode(raw).decode("utf-8"),
            "mime_type": "audio/wav",
            "sample_rate": int(sr),
            "duration_seconds": duration,
            "characters": len(text),
            "reference_bytes": ref_bytes,
            "mode": mode,
            "model": MODES[mode]["repo"],
            "model_revision": MODES[mode]["sha"],
            "license": MODEL_LICENSE,
            "weights_cache": _cache_kind,
            "cold_start": cold,
            "model_load_seconds": _load_seconds.get(mode, 0.0),
            "latency_seconds": round(time.time() - started, 2),
        }
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return {"error": type(exc).__name__ + ": " + str(exc)}


# Warm the DESIGN checkpoint during worker init. It is the one the lab reaches for first, and a long
# one-off load belongs in init rather than inside a request, where it competes with the execution timeout.
# The clone checkpoint loads on its first use.
#
# A failure here is printed in full and surfaced by the first request as a plain error, rather than
# leaving a worker that looks healthy and fails everything it touches.
try:
    _set_cache_env()
    _load("voice_design")
except Exception as _exc:  # noqa: BLE001
    _init_error = type(_exc).__name__ + ": " + str(_exc)
    print("[worker] INIT FAILED: " + _init_error, flush=True)
    traceback.print_exc()

runpod.serverless.start({"handler": handler})
