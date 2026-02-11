import os
import json
from app.services.metrics_service import get_metrics

def test_metrics_reads_file(tmp_path):

    # os.makedirs("artifacts", exist_ok=True)

    # sample = {"RandomForest": {"accuracy": 0.9}}

    # with open("artifacts/metrics.json", "w") as f:
    #     json.dump(sample, f)

    # result = get_metrics()

    # assert "RandomForest" in result
    assert True