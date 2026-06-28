import os
import tempfile
import threading
import requests

from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from google import genai
from supabase import create_client

app = FastAPI()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

genai_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY) if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY else None


class DriveJobRequest(BaseModel):
    job_id: str
    source_file_id: str


@app.get("/")
def health_check():
    return {"status": "ok", "service": "euro-scouting-worker-v2"}


@app.post("/process-drive-job")
def process_drive_job(req: DriveJobRequest, background_tasks: BackgroundTasks):
    if not genai_client:
        raise HTTPException(status_code=500, detail="Missing GEMINI_API_KEY")
    if not supabase:
        raise HTTPException(status_code=500, detail="Missing Supabase config")

    background_tasks.add_task(run_job, req.job_id, req.source_file_id)

    return {
        "accepted": True,
        "job_id": req.job_id,
        "source_file_id": req.source_file_id
    }


def update_job(job_id: str, data: dict):
    supabase.table("analysis_jobs").update(data).eq("id", job_id).execute()


def download_drive_file_stream(file_id: str, output_path: str):
    url = f"https://drive.google.com/uc?export=download&id={file_id}"

    with requests.Session() as session:
        response = session.get(url, stream=True)
        response.raise_for_status()

        with open(output_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)


def run_job(job_id: str, source_file_id: str):
    try:
        update_job(job_id, {
            "status": "processing",
            "error_message": None
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, "video.mp4")

            download_drive_file_stream(source_file_id, video_path)

            uploaded_file = genai_client.files.upload(file=video_path)

            prompt = """
Analyze this 7v7 football match video.

Return a structured scouting report in JSON with:
- match_summary
- tactical_observations
- key_events
- team_strengths
- team_weaknesses
- player_observations
- recommended_next_steps

If exact player identification is uncertain, say so clearly.
"""

            response = genai_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    prompt,
                    uploaded_file
                ]
            )

            result_text = response.text if hasattr(response, "text") else str(response)

            update_job(job_id, {
                "status": "completed",
                "gemini_file_uri": uploaded_file.uri,
                "ai_model": "gemini-2.5-flash",
                "raw_response": result_text
            })

    except Exception as e:
        update_job(job_id, {
            "status": "failed",
            "error_message": str(e)
        })
