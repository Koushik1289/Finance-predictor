# 🚀 Enterprise Finance Risk Analytics Platform

A production-ready Machine Learning platform for financial risk prediction built with:

- ⚡ FastAPI (Uvicorn Server)
- 🤖 Ensemble ML (RandomForest, XGBoost, LightGBM)
- 🎯 Optuna Hyperparameter Optimization
- 📊 Gradio Interactive Dashboard
- 🐘 PostgreSQL
- 🐳 Docker & Docker Compose
- 🔁 GitHub Actions CI/CD
- 🧪 Pytest with Coverage Control

---

# 📌 Project Overview

This system analyzes personal financial data and predicts **Financial Risk** using ensemble machine learning models.

## Risk Classification Logic

A user is classified as:

### 🔴 High Risk (1) if:
- Disposable Income < 0
- Disposable Income < Desired Savings
- Debt Ratio > 40%

Otherwise:

### 🟢 Low Risk (0)

---

# 🏗 System Architecture

FastAPI (Uvicorn)  
        ↓  
Service Layer  
        ↓  
ML Layer (Bagging + Boosting + Optuna)  
        ↓  
PostgreSQL  
        ↓  
Gradio Dashboard UI  

---

# 📂 Project Structure

app/  
&nbsp;&nbsp;&nbsp;&nbsp;api/  
&nbsp;&nbsp;&nbsp;&nbsp;core/  
&nbsp;&nbsp;&nbsp;&nbsp;services/  
&nbsp;&nbsp;&nbsp;&nbsp;database/  

ml/  
tests/  
data/  
artifacts/  
dashboard.py  
Dockerfile  
docker-compose.yml  
requirements.txt  
README.md  

---

# ⚙️ Local Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/yourusername/finance-risk-platform.git
cd finance-risk-platform
```

## 2️⃣ Create Virtual Environment

```bash
python -m venv venv
```

Activate:

Windows:
```bash
venv\Scripts\activate
```

Mac/Linux:
```bash
source venv/bin/activate
```

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

## 4️⃣ Add Dataset

Place dataset inside:

```
data/data.csv
```

---

# 🚀 Running the Backend (Uvicorn)

Start FastAPI server:

```bash
uvicorn app.main:app --reload
```

Server will run at:

```
http://127.0.0.1:8000
```

Swagger Documentation:

```
http://127.0.0.1:8000/docs
```

---

# ▶ Train the Model

```bash
curl -X POST http://127.0.0.1:8000/pipeline/run
```

This generates:

artifacts/  
- model.pkl  
- feature_columns.pkl  
- metrics.json  
- roc.png  
- comparison.png  

---

# ▶ Make Prediction

Example:

```bash
curl -X POST http://127.0.0.1:8000/predict \
-H "Content-Type: application/json" \
-d '{
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
}'
```

Example Response:

```json
{
  "prediction": 1,
  "risk_label": "High Risk",
  "confidence_score": 0.87
}
```

---

# 📊 Running Gradio Dashboard

⚠ Ensure FastAPI server is running first.

Run:

```bash
python dashboard.py
```

Dashboard will open at:

```
http://127.0.0.1:7860
```

Features:
- Train Model Button
- View Metrics
- Accuracy Comparison Chart
- ROC Curve Visualization
- Interactive Risk Prediction Form

---

# 🐳 Docker Usage

## Build Image

```bash
docker build -t finance-risk-platform .
```

## Run Container

```bash
docker run -p 8000:8000 finance-risk-platform
```

---

# 🐘 Docker Compose (With PostgreSQL)

```bash
docker compose up
```

---

# 🧪 Running Tests

Run with coverage (minimum 60%):

```bash
pytest --cov=app --cov=ml --cov-report=term --cov-fail-under=60
```

---

# 🔁 CI/CD Pipeline (GitHub Actions)

On every push:

- Ruff lint with `--fix`
- Pytest execution
- Coverage enforcement (≥ 60%)
- Docker image build
- Push to Docker Hub

Docker Hub Image:

```
spoorthi225f2/finance-risk-platform:latest
```

---

# 🔐 Environment Variables

Create `.env` file:

```
POSTGRES_DB=mydb
POSTGRES_USER=user
POSTGRES_PASSWORD=userpassword
```

Never commit `.env`.

---

# 🔥 Key Features

✔ Ensemble Machine Learning  
✔ Optuna Hyperparameter Optimization  
✔ Automatic Best Model Selection  
✔ Financial Risk Classification  
✔ Feature Engineering (Debt Ratio)  
✔ PostgreSQL Integration  
✔ Gradio Dashboard  
✔ Dockerized Deployment  
✔ GitHub CI/CD  
✔ Coverage Enforcement  

---

# 🛡 Security Notes

- Do NOT commit `.env`
- Do NOT commit `artifacts/`
- Use GitHub Secrets for Docker credentials
- Change exposed passwords immediately

---