from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import image_edit
from app.core.config.config import get_settings
from app.utils.logging import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)
settings = get_settings()

app = FastAPI(title="AI Image Editor API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(image_edit.router, prefix="/api/v1")


@app.on_event("startup")
async def startup_event() -> None:
    logger.info(
        "app_startup",
        extra={
            "model_device": settings.model_device,
            "model_id": settings.model_id,
            "max_parallel_jobs": settings.max_parallel_jobs,
        },
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
