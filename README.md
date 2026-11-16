# ClipMaster AI — AI-assisted video clipping

## Included
- FastAPI backend and responsive web interface.
- Upload local MP4/MOV/MKV/WebM videos.
- Faster-Whisper speech transcription (CPU/int8).
- Transcript-based highlight ranking using speech density and emphasis words.
- 15–60 second clip exports, vertical 9:16 / square / landscape.
- Word-timed caption groups burned into clips with four visual presets.
- FFmpeg video processing.

## Important limitations
This is an AI-assisted prototype, not a guaranteed viral-moment detector. It ranks windows using transcript/speech heuristics; it does not understand the full story or guarantee that selected moments are the best. Review all clips before publishing.
A YouTube URL alone is not processed. Upload the original video file you own or are authorised to edit. This avoids relying on unofficial YouTube downloading.
First run downloads the Whisper model (default `base`) from Hugging Face; hosting needs outbound access and enough disk/RAM. You can set `WHISPER_MODEL=tiny` for a smaller/faster model or `small` for potentially better transcription at higher resource cost.
Free hosting may sleep, time out on long videos, or lack sufficient memory/CPU. There is no honest way to promise unlimited, always-on, zero-cost video processing. A Docker-capable host is required; static hosting is not enough.

## Run locally
Install FFmpeg and Python 3.11+, then:
```bash
pip install -r requirements.txt
uvicorn app:app --reload
```
Open http://127.0.0.1:8000

## Deploy
Deploy as a Docker web service on a Docker-capable host. Expose the `PORT` environment variable (default 8000). Persistent disk is recommended for outputs. For a public deployment, add authentication, upload-size limits, rate limiting, automatic deletion of old clips, and private storage. This starter does not include user accounts or quotas.
