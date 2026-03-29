from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import logging

from app.routers.routes import router
from config import LOG_FILE

# Logging setup
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

app = FastAPI(
    title="Remediation AI Assistant",
    description="AI-powered remediation search tool",
    version="1.0.0"
    )

app.mount("/static", StaticFiles(directory="static"), name="static")

app.include_router(router)