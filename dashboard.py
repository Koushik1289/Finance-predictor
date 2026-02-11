import gradio as gr
import requests
from PIL import Image
import os

# Change port if needed
API = "http://127.0.0.1:8000"


# -------------------------
# API FUNCTIONS
# -------------------------

def train_model():
    try:
        response = requests.post(f"{API}/pipeline/run")
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def get_metrics():
    try:
        response = requests.get(f"{API}/model/metrics")
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def show_comparison_chart():
    if os.path.exists("artifacts/comparison.png"):
        return Image.open("artifacts/comparison.png")
    return None


def show_roc_chart():
    if os.path.exists("artifacts/roc.png"):
        return Image.open("artifacts/roc.png")
    return None


def predict(
    Income,
    Age,
    Dependents,
    Rent,
    Loan_Repayment,
    Insurance,
    Groceries,
    Transport,
    Eating_Out,
    Entertainment,
    Utilities,
    Healthcare,
    Education,
    Miscellaneous,
    Disposable_Income,
    Desired_Savings,
    Occupation,
    City_Tier
):
    payload = {
        "Income": Income,
        "Age": Age,
        "Dependents": Dependents,
        "Rent": Rent,
        "Loan_Repayment": Loan_Repayment,
        "Insurance": Insurance,
        "Groceries": Groceries,
        "Transport": Transport,
        "Eating_Out": Eating_Out,
        "Entertainment": Entertainment,
        "Utilities": Utilities,
        "Healthcare": Healthcare,
        "Education": Education,
        "Miscellaneous": Miscellaneous,
        "Disposable_Income": Disposable_Income,
        "Desired_Savings": Desired_Savings,
        "Occupation": Occupation,
        "City_Tier": City_Tier,
    }

    try:
        response = requests.post(f"{API}/predict", json=payload)
        return response.json()
    except Exception as e:
        return {"error": str(e)}


# -------------------------
# GRADIO UI
# -------------------------

with gr.Blocks() as demo:
    gr.Markdown("# 🚀 Enterprise Finance Risk ML Dashboard")

    with gr.Tab("Training & Metrics"):

        output = gr.JSON(label="Output")

        gr.Button("Train Model").click(train_model, None, output)
        gr.Button("View Metrics").click(get_metrics, None, output)
        gr.Button("Show Accuracy Comparison").click(
            show_comparison_chart, None, gr.Image()
        )
        gr.Button("Show ROC Curve").click(
            show_roc_chart, None, gr.Image()
        )

    with gr.Tab("Prediction"):

        Income = gr.Number(label="Income")
        Age = gr.Number(label="Age")
        Dependents = gr.Number(label="Dependents")
        Rent = gr.Number(label="Rent")
        Loan_Repayment = gr.Number(label="Loan Repayment")
        Insurance = gr.Number(label="Insurance")
        Groceries = gr.Number(label="Groceries")
        Transport = gr.Number(label="Transport")
        Eating_Out = gr.Number(label="Eating Out")
        Entertainment = gr.Number(label="Entertainment")
        Utilities = gr.Number(label="Utilities")
        Healthcare = gr.Number(label="Healthcare")
        Education = gr.Number(label="Education")
        Miscellaneous = gr.Number(label="Miscellaneous")
        Disposable_Income = gr.Number(label="Disposable Income")
        Desired_Savings = gr.Number(label="Desired Savings")

        Occupation = gr.Dropdown(
            ["Salaried", "Self_Employed", "Business"],
            label="Occupation"
        )

        City_Tier = gr.Dropdown(
            ["Tier_1", "Tier_2", "Tier_3"],
            label="City Tier"
        )

        prediction_output = gr.JSON(label="Prediction Result")

        gr.Button("Predict Risk").click(
            predict,
            [
                Income,
                Age,
                Dependents,
                Rent,
                Loan_Repayment,
                Insurance,
                Groceries,
                Transport,
                Eating_Out,
                Entertainment,
                Utilities,
                Healthcare,
                Education,
                Miscellaneous,
                Disposable_Income,
                Desired_Savings,
                Occupation,
                City_Tier
            ],
            prediction_output
        )

demo.launch(server_port=7860)
