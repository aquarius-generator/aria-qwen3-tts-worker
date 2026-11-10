# aria-qwen3-tts-worker

RunPod Serverless worker running **Qwen3-TTS** voice cloning, built for ARIA's `model-lab` blind voice
comparison.

**Licence: Apache-2.0**, confirmed from the Hugging Face API. Weights are 4.52 GB.

## Why this is its own image

`qwen-tts` pins `transformers==4.57.3`; the sibling Chatterbox worker's package pins `5.2.0`. They are
mutually exclusive, so one image could never hold both.

## The one asymmetry, and it is the model's

Qwen clones from the reference clip **and its transcript** (`ref_audio` plus `ref_text`). Chatterbox needs
only the clip. So `reference_text` is required here, and a missing one is a hard error rather than an empty
string passed through: cloning against a wrong transcript degrades the voice in a way that is easy to
mistake for the model simply being worse than its rival.

## No flash attention

The model card example passes `attn_implementation="flash_attention_2"`. That pulls `flash-attn`, a long
source build which would put this image back under the 1800 second build ceiling it is written to stay
inside, and buys nothing for single short utterances.

## Contract

**Input**

```json
{ "input": {
  "text": "required",
  "reference_audio": { "base64": "...", "content_type": "audio/wav" },
  "reference_text": "required, the transcript of the clip",
  "language": "English",
  "seed": 12
}}
```

**Output**: identical in shape to the Chatterbox worker.

```json
{
  "audio_base64": "...", "mime_type": "audio/wav", "sample_rate": 24000,
  "duration_seconds": 12.3, "model": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
  "license": "Apache-2.0", "weights_cache": "network-volume", "latency_seconds": 4.2
}
```

## Deploying

Connect GitHub in the RunPod console, deploy this repository, attach the shared network volume, and set
**min workers 0, max workers 1**. The endpoint id goes to ARIA as `MODEL_LAB_VOICE_B_ENDPOINT`.

## Licence

Worker code MIT. Model weights Apache-2.0 per the upstream model card.
