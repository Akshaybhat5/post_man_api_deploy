#lightweight dependencies
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np

class ScoringItem(BaseModel):
    distance_from_store: float #1.0, //float value,
    gender: str #"F", //str,
    credit_score: float #3.0, //float
    total_sales: float #4.0, //float
    total_items: int #43, //int
    transaction_count: int #2, //int
    product_area_id_count: int #1 //int

with open("example_lin_model_pipeline.pkl", "rb") as pipeline:
    full_pipeline = pickle.load(pipeline)

with open("example_lin_model.pkl", "rb") as lin_reg:
    model = pickle.load(lin_reg)

#app
app = FastAPI()

@app.get("/")
async def entry_point():
    return {"hello": "world"}


@app.post("/")
async def scoring_endpoint(item: ScoringItem):
    df = pd.DataFrame([item.dict()])
    df_preprocessed = full_pipeline.transform(df)
    yhat = model.predict(df_preprocessed)
    yhat = yhat.tolist() if isinstance(yhat, np.ndarray) else yhat[0]
    yhat = yhat[0]
    return {"prediction": yhat}