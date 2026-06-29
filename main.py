import os
import json
import time
import gc
import tempfile
import subprocess
from typing import Any, Dict, List

import gdown
from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
from supabase import create_client

app = FastAPI(title="Euro Scouting Worker", version="3.3-low-memory")

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
    return {"status": "ok", "service": "euro-scouting-worker-v3.3-low-memory"}


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


def split_drive_video(file_id: str, output_dir: str, segment_seconds: int = 180) -> List[str]:
    source_path = os.path.join(output_dir, "_source_tmp.mp4")

    print("Downloading source video...", flush=True)
    result = gdown.download(id=file_id, output=source_path, quiet=False)

    if not result or not os.path.exists(source_path) or os.path.getsize(source_path) == 0:
        raise RuntimeError("Google Drive download failed. Check sharing: Anyone with the link -> Viewer.")

    print(f"Downloaded file size: {os.path.getsize(source_path)} bytes", flush=True)

    output_pattern = os.path.join(output_dir, "chunk_%03d.mp4")

    print("Splitting video into small chunks...", flush=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-i", source_path,
        "-map", "0:v:0",
        "-map", "0:a:0?",
        "-c", "copy",
        "-f", "segment",
        "-segment_time", str(segment_seconds),
        "-reset_timestamps", "1",
        output_pattern
    ]

    subprocess.run(cmd, check=True)

    try:
        os.remove(source_path)
    except Exception:
        pass

    gc.collect()

    chunks = sorted(
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if f.startswith("chunk_") and f.endswith(".mp4")
    )

    if not chunks:
        raise RuntimeError("No chunks were created by ffmpeg.")

    print(f"Created {len(chunks)} chunks", flush=True)
    return chunks


def wait_for_gemini_file_active(uploaded_file: Any, timeout_seconds: int = 600):
    print("Waiting for Gemini file to become ACTIVE...", flush=True)

    started_at = time.time()

    while time.time() - started_at < timeout_seconds:
        file_state = genai_client.files.get(name=uploaded_file.name)
        state_name = getattr(file_state.state, "name", str(file_state.state))

        print(f"Gemini file state: {state_name}", flush=True)

        if state_name == "ACTIVE":
            return file_state

        if state_name == "FAILED":
            raise RuntimeError("Gemini file processing failed")

        time.sleep(5)

    raise RuntimeError("Gemini file did not become ACTIVE in time")


def delete_gemini_file(uploaded_file: Any):
    try:
        if uploaded_file and getattr(uploaded_file, "name", None):
            genai_client.files.delete(name=uploaded_file.name)
            print("Deleted Gemini uploaded file", flush=True)
    except Exception as e:
        print(f"Could not delete Gemini file: {e}", flush=True)


def analyze_chunk(chunk_path: str, chunk_index: int, chunks_count: int):
    uploaded_file = None

    try:
        print(f"Uploading chunk {chunk_index}/{chunks_count} to Gemini...", flush=True)

        uploaded_file = genai_client.files.upload(file=chunk_path)
        uploaded_file = wait_for_gemini_file_active(uploaded_file)

        prompt = f"""
You are a professional 7v7 football video analyst.

Analyze chunk {chunk_index} of {chunks_count} from a 7v7 football match.

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
  "chunks_count": {chunks_count},
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

        text = getattr(response, "text", str(response))

        return {
            "chunk_index": chunk_index,
            "gemini_file_uri": getattr(uploaded_file, "uri", None),
            "text": text
        }

    finally:
        delete_gemini_file(uploaded_file)

        try:
            os.remove(chunk_path)
        except Exception:
            pass

        del uploaded_file
        gc.collect()


def run_job(job_id: str, source_file_id: str):
    chunk_results = []

    try:
        print(f"Job started: {job_id}", flush=True)

        update_job(job_id, {
            "status": "running",
            "error_message": None,
            "result_json": {
                "status": "running",
                "chunks": []
            }
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            chunks = split_drive_video(source_file_id, tmpdir, segment_seconds=180)
            chunks_count = len(chunks)

            for i, chunk in enumerate(chunks, start=1):
                print(f"Analyzing chunk {i}/{chunks_count}", flush=True)

                result = analyze_chunk(chunk, i, chunks_count)
                chunk_results.append(result)

                update_job(job_id, {
                    "status": "running",
                    "result_json": {
                        "status": "running",
                        "chunks_count": chunks_count,
                        "processed_chunks": i,
                        "chunks": chunk_results
                    },
                    "raw_response": {
                        "text": json.dumps(chunk_results, ensure_ascii=False)
                    }
                })

                gc.collect()

            update_job(job_id, {
                "status": "completed",
                "ai_model": "gemini-2.5-flash",
                "result_json": {
                    "status": "completed",
                    "chunks_count": chunks_count,
                    "processed_chunks": chunks_count,
                    "chunks": chunk_results
                },
                "raw_response": {
                    "text": json.dumps(chunk_results, ensure_ascii=False)
                }
            })

        print(f"Job completed: {job_id}", flush=True)

    except Exception as e:
        print(f"Job failed: {job_id} | {str(e)}", flush=True)

        update_job(job_id, {
            "status": "failed",
            "error_message": str(e),
            "result_json": {
                "status": "failed",
                "partial_chunks": chunk_results
            }
        })

    finally:
        gc.collect()
