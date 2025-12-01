import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends, Form, Header, HTTPException, UploadFile, status
from pydantic import HttpUrl

from app.core.config.config import Settings, get_settings
from app.services.image_processor import ImageEditJob, handle_image_edit
from app.utils.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)
_semaphore = asyncio.Semaphore(get_settings().max_parallel_jobs)


def verify_api_key(
    x_api_key: Annotated[str | None, Header(convert_underscores=False)] = None,
    settings: Settings = Depends(get_settings),
) -> None:
    if not x_api_key or x_api_key != settings.api_key:
        logger.warning("auth_failed", extra={"provided": x_api_key})
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")


async def _schedule_job(job: ImageEditJob) -> None:
    async with _semaphore:
        await handle_image_edit(job)


@router.post("/image/edit", status_code=status.HTTP_202_ACCEPTED)
async def edit_image(
    image: UploadFile,
    prompt: Annotated[str, Form(...)],
    webhook_url: Annotated[HttpUrl, Form(...)],
    _: None = Depends(verify_api_key),
    settings: Settings = Depends(get_settings),
) -> dict[str, str]:
    content = await image.read()
    job = ImageEditJob(prompt=prompt, image_bytes=content, webhook_url=str(webhook_url), settings=settings)

    asyncio.create_task(_schedule_job(job))
    logger.info(
        "job_enqueued",
        extra={
            "job_id": job.job_id,
            "filename": image.filename,
            "webhook": str(webhook_url),
            "prompt": prompt,
        },
    )
    return {"job_id": job.job_id, "status": "queued"}
