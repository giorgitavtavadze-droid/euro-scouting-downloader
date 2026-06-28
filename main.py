import os
import time
import json
import tempfile
import subprocess
from typing import Any, Dict, List

import gdown
from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
from supabase import create_client

app = FastAPI(title="Euro Scouting Worker", version="3.0")

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
    return {"status": "ok", "service": "euro-scouting-worker-v3"}


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


def update_job(job_id: str, data: Dict[str, Any]):
    supabase.table("analysis_jobs").update(data).eq("id", job_id).execute()


def download_drive_file(file_id: str, output_path: str):
    result = gdown.download(id=file_id, output=output_path, quiet=False)

    if not result or not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        raise RuntimeError("Google Drive download failed. Check sharing: Anyone with the link -> Viewer.")


def split_video(input_path: str, output_dir: str, segment_seconds: int = 600) -> List[str]:
    output_pattern = os.path.join(output_dir, "chunk_%03d.mp4")

    cmd = [
        "ffmpeg",
        "-i", input_path,
        "-c", "copy",
        "-map", "0",
        "-f", "segment",
        "-segment_time", str(segment_seconds),
        "-reset_timestamps", "1",
        output_pattern
    ]

    subprocess.run(cmd, check=True)

    chunks = sorted(
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if f.endswith(".mp4")
    )

    if not chunks:
        raise RuntimeError("No chunks were created by ffmpeg.")

    return chunks


def analyze_chunk(chunk_path: str, chunk_index: int):
    uploaded_file = genai_client.files.upload(file=chunk_path)

    prompt = f"""
You are a professional 7v7 football video analyst.

Analyze chunk #{chunk_index} of a football match video.

Return Georgian JSON only.

Focus on:
- visible score if shown
- key events
- player shirt numbers if visible
- player actions
- pressing
- defensive structure
- attacking structure
- transitions
- mistakes
- confidence level

If something is unclear, say confidence is low.

JSON:
{{
  "chunk_index": {chunk_index},
  "summary": "",
  "key_events": [],
  "player_observations": [],
  "tactical_notes": [],
  "confidence": "low|medium|high"
}}
"""

    response = genai_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[prompt, uploaded_file]
    )

    return {
        "chunk_index": chunk_index,
        "gemini_file_uri": getattr(uploaded_file, "uri", None),
        "text": getattr(response, "text", str(response))
    }


def run_job(job_id: str, source_file_id: str):
    try:
        update_job(job_id, {
            "status": "running",
            "error_message": None
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, "source_video.mp4")
            chunks_dir = os.path.join(tmpdir, "chunks")
            os.makedirs(chunks_dir, exist_ok=True)

            download_drive_file(source_file_id, video_path)
            chunks = split_video(video_path, chunks_dir, segment_seconds=600)

            chunk_results = []
            for i, chunk in enumerate(chunks, start=1):
                result = analyze_chunk(chunk, i)
                chunk_results.append(result)

            update_job(job_id, {
                "status": "completed",
                "ai_model": "gemini-2.5-flash",
                "result_json": {
                    "chunks_count": len(chunk_results),
                    "chunks": chunk_results
                },
                "raw_response": {
                    "text": json.dumps(chunk_results, ensure_ascii=False)
                }
            })

    except Exception as e:
        update_job(job_id, {
            "status": "failed",
            "error_message": str(e)
        })
