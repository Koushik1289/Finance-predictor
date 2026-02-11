from fastapi import FastAPI
from app.api import pipeline, predict, metrics, health

app = FastAPI()

app.include_router(health.router)
app.include_router(pipeline.router)
app.include_router(predict.router)
app.include_router(metrics.router)
