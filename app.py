import os, uuid, subprocess, shutil, re, math
from pathlib import Path
from collections import Counter
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

ROOT = Path(__file__).parent
UPLOADS, OUTPUTS, WORK = ROOT/"uploads", ROOT/"outputs", ROOT/"work"
for folder in (UPLOADS, OUTPUTS, WORK):
    folder.mkdir(exist_ok=True)

app = FastAPI(title="ClipMaster AI")
app.mount("/static", StaticFiles(directory=ROOT/"static"), name="static")
templates = Jinja2Templates(directory=str(ROOT/"templates"))

HYPE_WORDS = {
    "amazing","insane","crazy","secret","never","best","worst","shocking","truth",
    "finally","impossible","unbelievable","wow","mistake","million","billion",
    "danger","fail","success","win","won","lose","lost","revealed","surprise",
    "लेकिन","मगर","क्योंकि","सच","कमाल","जबरदस्त","शानदार","खतरनाक","जीत","हार",
    "राज","खुलासा","आखिर","पहली","पहला","देखिए","सोचिए","अविश्वसनीय"
}

def run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode:
        raise RuntimeError(p.stderr[-3000:])
    return p.stdout.strip()

def get_duration(path):
    val = run(["ffprobe","-v","error","-show_entries","format=duration",
               "-of","default=noprint_wrappers=1:nokey=1",str(path)])
    return float(val)

def transcribe(path):
    try:
        from faster_whisper import WhisperModel
    except Exception as e:
        raise RuntimeError("Speech recognition is unavailable. Check that faster-whisper installed correctly.")
    # CPU/int8 is the most compatible option for low-cost hosting.
    model_name = os.getenv("WHISPER_MODEL", "base")
    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    segments, info = model.transcribe(str(path), word_timestamps=True, vad_filter=True, beam_size=3)
    words = []
    for seg in segments:
        if seg.words:
            for w in seg.words:
                token = (w.word or "").strip()
                if token:
                    words.append({"start": float(w.start), "end": float(w.end), "text": token})
        elif seg.text.strip():
            words.append({"start": float(seg.start), "end": float(seg.end), "text": seg.text.strip()})
    return words, getattr(info, "language", "unknown")

def norm_word(word):
    return re.sub(r"[^\w\u0900-\u097F]+", "", word.lower())

def choose_highlights(words, total, clip_len):
    if total <= clip_len:
        return [(0.0, total)]
    # Score candidate windows on speech density and attention-grabbing language.
    # This is a lightweight transcript heuristic, not a human-level semantic judgement.
    step = 8
    candidates = []
    for start in range(0, max(1, int(total-clip_len)+1), step):
        end = min(total, start+clip_len)
        inside = [w for w in words if start <= w["start"] < end]
        tokens = [norm_word(w["text"]) for w in inside]
        tokens = [t for t in tokens if t]
        if not tokens:
            score = 0.0
        else:
            hype = sum(1 for t in tokens if t in HYPE_WORDS)
            punctuation = sum(1 for w in inside if any(c in w["text"] for c in "!?।"))
            speech_rate = len(tokens) / max(1, clip_len)
            # Favor lively spoken windows while not only selecting keyword-heavy spans.
            score = (min(len(tokens), 500) * 0.06) + hype * 2.2 + punctuation * 0.45 + min(speech_rate, 4) * 1.4
        candidates.append((score, float(start), float(end)))
    candidates.sort(reverse=True)
    selected = []
    for score, start, end in candidates:
        if all(end <= s or start >= e for s,e in selected):
            selected.append((start,end))
        if len(selected) >= max(1, min(12, math.ceil(total/clip_len))):
            break
    if not selected:
        selected = [(0.0, min(total, clip_len))]
    return sorted(selected)

def ass_time(seconds):
    h = int(seconds//3600); m = int((seconds%3600)//60)
    s = seconds%60
    return f"{h}:{m:02d}:{s:05.2f}"

def srt_time(seconds):
    ms = int(round(seconds*1000))
    h, ms = divmod(ms, 3600000); m, ms = divmod(ms, 60000); s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def write_captions(words, start, end, path, style):
    # Word-level timings are grouped into short readable phrases.
    subset = [w for w in words if w["end"] > start and w["start"] < end]
    groups, current = [], []
    for w in subset:
        current.append(w)
        txt = " ".join(x["text"] for x in current)
        if len(current) >= 4 or len(txt) >= 32 or any(c in w["text"] for c in ".!?।"):
            groups.append(current); current = []
    if current: groups.append(current)
    with open(path, "w", encoding="utf-8") as f:
        for i, group in enumerate(groups, 1):
            a = max(0, group[0]["start"]-start)
            b = min(end-start, group[-1]["end"]-start)
            text = " ".join(x["text"] for x in group).replace("-->", "→")
            f.write(f"{i}\n{srt_time(a)} --> {srt_time(max(a+0.25,b))}\n{text}\n\n")

def subtitle_filter(path):
    # ffmpeg subtitles filter requires escaping path punctuation.
    p = str(path).replace("\\","/").replace(":","\\:").replace("'","\\'")
    return f"subtitles='{p}':force_style='FontName=DejaVu Sans,FontSize=18,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=1,Outline=3,Shadow=1,Alignment=2,MarginV=90,Bold=1'"

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request":request})

@app.post("/api/process")
async def process_video(video: UploadFile = File(...), caption_style: str = Form("bold"),
                        clip_length: int = Form(60), aspect: str = Form("9:16")):
    if not video.filename:
        raise HTTPException(400, "Choose a video file.")
    ext = Path(video.filename).suffix.lower()
    if ext not in {".mp4",".mov",".mkv",".webm",".m4v"}:
        raise HTTPException(400, "Upload MP4, MOV, MKV, or WebM.")
    if clip_length < 15 or clip_length > 60:
        raise HTTPException(400, "Clip length must be between 15 and 60 seconds.")
    job = uuid.uuid4().hex[:10]
    source = UPLOADS/f"{job}{ext}"
    jobdir = WORK/job
    jobdir.mkdir(exist_ok=True)
    try:
        with source.open("wb") as f:
            shutil.copyfileobj(video.file, f)
        total = get_duration(source)
        if total < 3: raise HTTPException(400,"Video is too short.")
        words, language = transcribe(source)
        windows = choose_highlights(words, total, clip_length)
        clips = []
        for idx, (start,end) in enumerate(windows,1):
            srt = jobdir/f"clip_{idx:02d}.srt"
            write_captions(words,start,end,srt,caption_style)
            out = OUTPUTS/f"{job}_highlight_{idx:02d}.mp4"
            if aspect == "1:1":
                vf = "scale=1080:1080:force_original_aspect_ratio=increase,crop=1080:1080"
            elif aspect == "16:9":
                vf = "scale=1280:720:force_original_aspect_ratio=increase,crop=1280:720"
            else:
                vf = "scale=720:1280:force_original_aspect_ratio=increase,crop=720:1280"
            # Distinct caption presets via ASS/SRT base styling.
            if caption_style == "clean":
                style = "FontName=DejaVu Sans,FontSize=16,PrimaryColour=&H00FFFFFF,OutlineColour=&H80000000,BorderStyle=1,Outline=1,Shadow=0,Alignment=2,MarginV=75,Bold=0"
            elif caption_style == "classic":
                style = "FontName=DejaVu Sans,FontSize=18,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=1,Outline=2,Shadow=1,Alignment=2,MarginV=85,Bold=0"
            elif caption_style == "karaoke":
                style = "FontName=DejaVu Sans,FontSize=20,PrimaryColour=&H0000FFFF,OutlineColour=&H00000000,BorderStyle=1,Outline=3,Shadow=1,Alignment=2,MarginV=90,Bold=1"
            else:
                style = "FontName=DejaVu Sans,FontSize=21,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=1,Outline=4,Shadow=2,Alignment=2,MarginV=95,Bold=1"
            escaped = str(srt).replace("\\","/").replace(":","\\:").replace("'","\\'")
            vf_full = f"{vf},subtitles='{escaped}':force_style='{style}'"
            run(["ffmpeg","-y","-ss",str(start),"-i",str(source),"-t",str(end-start),
                 "-vf",vf_full,"-c:v","libx264","-preset","veryfast","-crf","24",
                 "-c:a","aac","-b:a","128k","-movflags","+faststart",str(out)])
            clips.append({"name":out.name,"url":f"/download/{out.name}","seconds":round(end-start),
                          "start":round(start,1),"end":round(end,1)})
        return JSONResponse({"ok":True,"message":f"Created {len(clips)} highlight clips.",
            "language":language,"clips":clips,
            "note":"Highlights are selected with transcript-based scoring (speech density and emphasis words). Review clips before publishing; automatic selection is approximate."})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500,str(e))
    finally:
        source.unlink(missing_ok=True)
        shutil.rmtree(jobdir,ignore_errors=True)

@app.get("/download/{filename}")
async def download(filename:str):
    path=OUTPUTS/Path(filename).name
    if not path.exists(): raise HTTPException(404,"Clip not found. Process the video again.")
    return FileResponse(path,media_type="video/mp4",filename=path.name)
