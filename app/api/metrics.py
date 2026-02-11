from fastapi import APIRouter
from app.services.metrics_service import get_metrics

router = APIRouter()

@router.get("/model/metrics")
def metrics():
    return get_metrics()