from fastapi import APIRouter
from app.services.pipeline_service import run_pipeline

router = APIRouter()

@router.post("/pipeline/run")
def run():
    return run_pipeline()