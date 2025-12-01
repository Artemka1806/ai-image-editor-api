"""
Example webhook receiver server.

Run locally:
    uvicorn webhook_server_example:app --reload --port 9000

The endpoint saves the uploaded image to ./webhook_uploads and logs the path.
"""

from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse

UPLOAD_DIR = Path("webhook_uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Webhook Receiver Example")


@app.post("/webhook")
async def receive_webhook(
    job_id: Annotated[str | None, Form(None)],
    status: Annotated[str | None, Form(None)],
    prompt: Annotated[str | None, Form(None)],
    image: UploadFile = File(...),
) -> JSONResponse:
    file_path = UPLOAD_DIR / f"{job_id or 'unknown'}_{image.filename}"
    content = await image.read()
    file_path.write_bytes(content)

    print(f"[webhook] job={job_id} status={status} prompt={prompt} saved={file_path}")
    return JSONResponse({"stored_path": str(file_path), "job_id": job_id, "status": status or "unknown"})
