# Qwen3-TTS (Apache-2.0) voice cloning as a RunPod Serverless worker.
#
# THE SAME FOUR GUARDS THE IMAGE WORKER COST FOUR BUILDS TO ESTABLISH, applied from the start:
#   1. No weights in the image. Baking them in blew the 1800 second build ceiling there.
#   2. Exact pins only. Ranged pins made pip backtrack and eat a whole build budget with no download at all.
#   3. The Python assert runs FIRST. A base shipping Python 3.10 killed a build nine minutes in, and the
#      build API returns only a buildx exit status with no log. Ten seconds beats nine minutes.
#   4. An import assert runs LAST. An image once built cleanly and then failed every job because the
#      library did not contain the class the model needed.
#
# WHY THIS IS ITS OWN IMAGE: qwen-tts pins transformers==4.57.3 and chatterbox-tts pins 5.2.0. Mutually
# exclusive, so one image could never hold both.
#
# NO FLASH ATTENTION. The model card's example passes attn_implementation="flash_attention_2", which pulls
# flash-attn, which is a long source build. It would put this image back under the ceiling it is written to
# stay inside, and buys nothing for single short utterances. The default attention is used instead.

FROM runpod/base:2.1.0-cuda1281-ubuntu2404

ENV PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_INPUT=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1

# The model and its licence, echoed on every response so the three can never silently disagree.
ENV VOICE_MODEL_ID="Qwen/Qwen3-TTS-12Hz-1.7B-Base" \
    VOICE_MODEL_LICENSE="Apache-2.0"

ENV VOICE_VOLUME_CACHE="/runpod-volume/hf" \
    VOICE_FALLBACK_CACHE="/app/hf-cache"

WORKDIR /app

# FIRST STEP ON PURPOSE. A wrong base fails here in seconds instead of nine minutes.
RUN python3 -c "import sys; print('python', sys.version); assert sys.version_info >= (3, 10), 'qwen-tts needs Python 3.10+; this base has ' + sys.version"

# libsndfile backs soundfile; sox is a real binary the qwen-tts package's python wrapper shells out to;
# ffmpeg decodes whatever reference clip arrives, since a caller may send mp3 or m4a rather than wav.
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 sox ffmpeg && rm -rf /var/lib/apt/lists/*

# torch first and alone, the largest single download, so editing any other pin does not re-fetch it.
RUN python3 -m pip install --no-cache-dir --prefer-binary \
        torch==2.6.0 torchaudio==2.6.0

COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install --no-cache-dir --prefer-binary -r /app/requirements.txt

# Fails the build here, cheaply, if what was just installed does not actually import.
RUN python3 -c "import torch, soundfile; from qwen_tts import Qwen3TTSModel; print('torch', torch.__version__, '| Qwen3TTSModel imported')"

COPY handler.py /app/handler.py

CMD ["python3", "-u", "/app/handler.py"]
