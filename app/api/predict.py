from fastapi import APIRouter
from pydantic import BaseModel
from app.services.prediction_service import predict

router = APIRouter()

class Request(BaseModel):
    Income:float
    Age:int
    Dependents:int
    Rent:float
    Loan_Repayment:float
    Insurance:float
    Groceries:float
    Transport:float
    Eating_Out:float
    Entertainment:float
    Utilities:float
    Healthcare:float
    Education:float
    Miscellaneous:float
    Disposable_Income:float
    Desired_Savings:float
    Occupation:str
    City_Tier:str

@router.post("/predict")
def run(req:Request):
    return predict(req.model_dump())