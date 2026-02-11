import pandas as pd
from app.core.feature_engineering import preprocess
from ml.train import train_all

def run_pipeline():
    df = pd.read_csv("data/data.csv")
    df = preprocess(df)
    return train_all(df)