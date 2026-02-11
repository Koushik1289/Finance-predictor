import os
import pandas as pd

os.environ["TESTING"] = "1"

from ml.train import train_all
from app.core.feature_engineering import preprocess


def test_train_logic_runs():
    df = pd.DataFrame({
        "Income": [50000, 40000],
        "Age": [30, 28],
        "Dependents": [2, 1],
        "Rent": [10000, 20000],
        "Loan_Repayment": [5000, 10000],
        "Insurance": [2000, 3000],
        "Groceries": [4000, 6000],
        "Transport": [2000, 3000],
        "Eating_Out": [1000, 2000],
        "Entertainment": [1500, 2000],
        "Utilities": [2500, 3000],
        "Healthcare": [1000, 2000],
        "Education": [2000, 3000],
        "Miscellaneous": [1000, 2000],
        "Disposable_Income": [5000, -2000],
        "Desired_Savings": [3000, 7000],
        "Occupation": ["Salaried", "Salaried"],
        "City_Tier": ["Tier_1", "Tier_1"],
    })

    df = preprocess(df)

    result = train_all(df)

    assert "accuracy" in result