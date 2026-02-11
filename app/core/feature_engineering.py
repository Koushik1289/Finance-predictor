import pandas as pd


def preprocess(df: pd.DataFrame):

    # Debt ratio feature
    df["Debt_Ratio"] = (
        (df["Loan_Repayment"] + df["Rent"]) / df["Income"]
    )

    # Improved Financial Risk Logic
    df["Financial_Risk"] = (
        (df["Disposable_Income"] < 0) |
        (df["Disposable_Income"] < df["Desired_Savings"]) |
        (df["Debt_Ratio"] > 0.4)
    ).astype(int)

    df = pd.get_dummies(
        df,
        columns=["Occupation", "City_Tier"],
        drop_first=True
    )

    return df
