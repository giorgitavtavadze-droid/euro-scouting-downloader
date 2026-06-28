import os
import tempfile
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from yt_dlp import YoutubeDL
from google import genai

app = FastAPI()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    client = genai.Client(api_key=GEMINI_API_KEY)
else:
    client = None


class ProcessVideoRequest(BaseModel):
    youtube_url: str


@app.get("/")
def health_check():
    return {"status": "ok", "service": "euro-scouting-downloader"}


@app.post("/process-video")
def process_video(req: ProcessVideoRequest):
    if not client:
        raise HTTPException(status_code=500, detail="Missing GEMINI_API_KEY")

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_template = os.path.join(tmpdir, "video.%(ext)s")

            ydl_opts = {
                "format": "best[ext=mp4]/best",
                "outtmpl": output_template,
                "quiet": True,
                "noplaylist": True,
                "max_filesize": 2 * 1024 * 1024 * 1024
            }

            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(req.youtube_url, download=True)
                file_path = ydl.prepare_filename(info)

            uploaded_file = client.files.upload(file=file_path)

            return {
                "status": "success",
                "youtube_url": req.youtube_url,
                "title": info.get("title"),
                "duration_seconds": info.get("duration"),
                "file_uri": uploaded_file.uri,
                "file_name": uploaded_file.name,
                "mime_type": uploaded_file.mime_type
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
