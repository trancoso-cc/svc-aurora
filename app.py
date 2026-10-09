import mlflow
from fastapi import FastAPI
MODEL_URI = "models:/m-8842/5"            # consumes (== pipeline-borealis provides)   [T2]
SCORING_ENDPOINT = "https://scoring-7f3a.a.run.app/predict"     # provides (== nightly-reconciler consumes)  [T1]
model = mlflow.pyfunc.load_model(MODEL_URI)
app = FastAPI()
@app.post("/predict")
def predict(x: dict): return {"p": float(model.predict([x])[0])}
