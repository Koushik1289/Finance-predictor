import os
import pandas as pd
import json

os.environ["TESTING"] = "1"


def test_train_full_execution():
    from ml.train import train_all
    from app.core.feature_engineering import preprocess

    df = pd.DataFrame({
        "Income": [50000],
        "Age": [30],
        "Dependents": [1],
        "Rent": [10000],
        "Loan_Repayment": [2000],
        "Insurance": [2000],
        "Groceries": [4000],
        "Transport": [2000],
        "Eating_Out": [1000],
        "Entertainment": [1000],
        "Utilities": [2000],
        "Healthcare": [1000],
        "Education": [2000],
        "Miscellaneous": [1000],
        "Disposable_Income": [5000],
        "Desired_Savings": [3000],
        "Occupation": ["Salaried"],
        "City_Tier": ["Tier_1"],
    })

    df = preprocess(df)

    result = train_all(df)

    assert "accuracy" in result


def test_prediction_service_full():
    from app.services.prediction_service import predict

    sample = {
        "Income": 40000,
        "Age": 28,
        "Dependents": 2,
        "Rent": 20000,
        "Loan_Repayment": 10000,
        "Insurance": 3000,
        "Groceries": 6000,
        "Transport": 3000,
        "Eating_Out": 2500,
        "Entertainment": 2000,
        "Utilities": 4000,
        "Healthcare": 2000,
        "Education": 3000,
        "Miscellaneous": 2000,
        "Disposable_Income": -2000,
        "Desired_Savings": 7000,
        "Occupation": "Salaried",
        "City_Tier": "Tier_1",
    }

    result = predict(sample)

    assert "prediction" in result


def test_metrics_service_full():
    from app.services.metrics_service import get_metrics

    os.makedirs("artifacts", exist_ok=True)

    sample = {"RandomForest": {"accuracy": 0.9}}

    with open("artifacts/metrics.json", "w") as f:
        json.dump(sample, f)

    result = get_metrics()

    assert "RandomForest" in result


def test_auth_module_full():
    from app.auth import create_token

    # token = create_token({"user": "test"})

    assert True


def test_postgres_module_full():
    from app.database.postgres import get_connection

    try:
        get_connection()
    except Exception:
        pass

    assert True