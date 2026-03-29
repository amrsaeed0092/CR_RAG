from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
import os
import uuid
import logging

from app.services.rag_service import run_rag

router = APIRouter()
templates = Jinja2Templates(directory="templates")

logger = logging.getLogger(__name__)

TEMP_DIR = "temp_outputs"
os.makedirs(TEMP_DIR, exist_ok=True)


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
    request=request, 
    name="index.html",
    context={"error": "Model decommissioned. Please update MODEL_NAME."}
)


@router.post("/run", response_class=HTMLResponse)
async def run_analysis(request: Request, issue: str = Form(...)):
    try:
        logger.info("Received new analysis request")
        df = run_rag(issue)

        file_id = str(uuid.uuid4())
        file_path = os.path.join(TEMP_DIR, f"{file_id}.xlsx")

        # This will now work once openpyxl is installed
        df.to_excel(file_path, index=False)
        table_html = df.to_html(classes="table", index=False)

        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "table": table_html,
                "file_id": file_id
            }
        )

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={"message": f"Error: {str(e)}"}
        )


@router.get("/download/{file_id}")
async def download(file_id: str):
    file_path = os.path.join(TEMP_DIR, f"{file_id}.xlsx")
    return FileResponse(
        path=file_path,
        filename="remediation_output.xlsx",
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )