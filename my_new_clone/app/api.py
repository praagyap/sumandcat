# api.py
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd

from inference import TransactionClassifier
from summarizer import simple_summarizer

app = FastAPI()
classifier = TransactionClassifier()

class TransactionInput(BaseModel):
    transactions: List[str]

class SummaryInput(BaseModel):
    transactions: List[str]
    categories: List[str]
    amounts: List[float]

@app.post("/classify")
def classify_transactions(data: TransactionInput):
    results = classifier.categorize(data.transactions)
    return {"categories": results}

@app.post("/summarize")
def summarize_transactions(data: SummaryInput):
    df = pd.DataFrame({
        "Transaction": data.transactions,
        "Category": data.categories,
        "Amount": data.amounts
    })
    summary = simple_summarizer(df)
    return {"summary": summary}
