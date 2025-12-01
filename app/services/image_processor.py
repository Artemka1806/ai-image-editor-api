import asyncio
import io
from pathlib import Path
from typing import Optional
from uuid import uuid4

import httpx
import torch
from diffusers import QwenImageEditPlusPipeline
from PIL import Image

from app.core.config.config import Settings
from app.utils.logging import get_logger

logger = get_logger(__name__)
_pipeline: QwenImageEditPlusPipeline | None = None
_pipeline_lock = asyncio.Lock()


class ImageEditJob:
    def __init__(
        self,
        *,
        prompt: str,
        image_bytes: bytes,
        webhook_url: str,
        settings: Settings,
        job_id: Optional[str] = None,
    ) -> None:
        self.job_id = job_id or str(uuid4())
        self.prompt = prompt
        self.image_bytes = image_bytes
        self.webhook_url = webhook_url
        self.settings = settings

    @property
    def job_dir(self) -> Path:
        return self.settings.storage_dir / self.job_id

    @property
    def input_path(self) -> Path:
        return self.job_dir / "input.png"

    @property
    def output_path(self) -> Path:
        return self.job_dir / "output.png"


async def _get_pipeline(settings: Settings) -> QwenImageEditPlusPipeline:
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    async with _pipeline_lock:
        if _pipeline is not None:
            return _pipeline

        logger.info(
            "pipeline_loading",
            extra={"model": settings.model_id, "device": settings.model_device},
        )

        def _load() -> QwenImageEditPlusPipeline:
            pipeline = QwenImageEditPlusPipeline.from_pretrained(
                settings.model_id, torch_dtype=torch.bfloat16
            )
            pipeline.to(settings.model_device)
            pipeline.set_progress_bar_config(disable=None)
            return pipeline

        _pipeline = await asyncio.to_thread(_load)
        logger.info("pipeline_loaded", extra={"model": settings.model_id})
        return _pipeline


async def handle_image_edit(job: ImageEditJob) -> None:
    logger.info(
        "job_started",
        extra={"job_id": job.job_id, "webhook": job.webhook_url, "prompt": job.prompt},
    )
    job.job_dir.mkdir(parents=True, exist_ok=True)
    await asyncio.to_thread(job.input_path.write_bytes, job.image_bytes)

    try:
        edited_image_bytes = await generate_image(job)
        await asyncio.to_thread(job.output_path.write_bytes, edited_image_bytes)
        await send_webhook(job, edited_image_bytes)
        logger.info(
            "job_completed",
            extra={
                "job_id": job.job_id,
                "webhook": job.webhook_url,
                "output_path": str(job.output_path),
            },
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("job_failed", extra={"job_id": job.job_id})
        await notify_failure(job, exc)


async def generate_image(job: ImageEditJob) -> bytes:
    pipeline = await _get_pipeline(job.settings)
    logger.debug(
        "render_start",
        extra={"job_id": job.job_id, "device": job.settings.model_device, "model": job.settings.model_id},
    )

    def _infer() -> bytes:
        input_image = Image.open(io.BytesIO(job.image_bytes)).convert("RGB")
        generator = torch.Generator(device=pipeline.device).manual_seed(
            int(uuid4().int % 2**32)
        )
        result = pipeline(
            image=input_image,
            prompt=job.prompt,
            negative_prompt=job.settings.default_negative_prompt,
            generator=generator,
            true_cfg_scale=4.0,
            num_inference_steps=40,
            guidance_scale=1.0,
            num_images_per_prompt=1,
        )
        output_image = result.images[0]
        buffer = io.BytesIO()
        output_image.save(buffer, format="PNG")
        return buffer.getvalue()

    image_bytes = await asyncio.to_thread(_infer)
    logger.debug("render_end", extra={"job_id": job.job_id})
    return image_bytes


async def send_webhook(job: ImageEditJob, image_bytes: bytes) -> None:
    payload = {"job_id": job.job_id, "status": "completed", "prompt": job.prompt}
    logger.info("webhook_dispatch", extra={"job_id": job.job_id, "webhook": job.webhook_url})
    async with httpx.AsyncClient(verify=job.settings.webhook_verify_ssl) as client:
        response = await client.post(
            job.webhook_url,
            data=payload,
            files={"image": ("result.png", image_bytes, "image/png")},
            timeout=job.settings.webhook_timeout_seconds,
        )
        response.raise_for_status()
    logger.info("webhook_ok", extra={"job_id": job.job_id, "status_code": response.status_code})


async def notify_failure(job: ImageEditJob, exc: Exception) -> None:
    try:
        async with httpx.AsyncClient(verify=job.settings.webhook_verify_ssl) as client:
            await client.post(
                job.webhook_url,
                json={
                    "job_id": job.job_id,
                    "status": "failed",
                    "reason": str(exc),
                },
                timeout=job.settings.webhook_timeout_seconds,
            )
    except Exception:  # noqa: BLE001
        logger.exception("webhook_failure", extra={"job_id": job.job_id})
