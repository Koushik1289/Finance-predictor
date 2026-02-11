import os
os.environ["TESTING"] = "1"

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_pipeline_runs():
    response = client.post("/pipeline/run")
    assert response.status_code == 200
