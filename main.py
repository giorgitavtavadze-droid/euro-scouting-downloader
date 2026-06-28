import os
import tempfile
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
import gdown

app = FastAPI()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None


class DriveJobRequest(BaseModel):
    job_id: str
    source_file_id: str


@app.get("/")
def health_check():
    return {"status": "ok", "service": "euro-scouting-downloader"}


@app.post("/process-drive-job")
def process_drive_job(req: DriveJobRequest):
    if not client:
        raise HTTPException(status_code=500, detail="Missing GEMINI_API_KEY")

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "video.mp4")

            drive_url = f"https://drive.google.com/uc?id={req.source_file_id}"
            gdown.download(drive_url, file_path, quiet=False)

            uploaded_file = client.files.upload(file=file_path)

            return {
                "status": "success",
                "job_id": req.job_id,
                "source_file_id": req.source_file_id,
                "file_uri": uploaded_file.uri,
                "file_name": uploaded_file.name,
                "mime_type": uploaded_file.mime_type
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
