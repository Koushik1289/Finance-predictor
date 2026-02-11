import json

def get_metrics():
    with open("artifacts/metrics.json") as f:
        return json.load(f)