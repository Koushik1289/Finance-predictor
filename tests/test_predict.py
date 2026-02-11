import os
os.environ["TESTING"] = "1"

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

low_risk_payload = {
    "Income": 80000,
    "Age": 30,
    "Dependents": 1,
    "Rent": 10000,
    "Loan_Repayment": 2000,
    "Insurance": 2000,
    "Groceries": 4000,
    "Transport": 2000,
    "Eating_Out": 1000,
    "Entertainment": 1000,
    "Utilities": 2000,
    "Healthcare": 1000,
    "Education": 2000,
    "Miscellaneous": 1000,
    "Disposable_Income": 20000,
    "Desired_Savings": 5000,
    "Occupation": "Salaried",
    "City_Tier": "Tier_1"
}

high_risk_payload = {
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
    "City_Tier": "Tier_1"
}


def test_low_risk_prediction():
    response = client.post("/predict", json=low_risk_payload)
    assert response.status_code == 200


def test_high_risk_prediction():
    response = client.post("/predict", json=high_risk_payload)
    assert response.status_code == 200
