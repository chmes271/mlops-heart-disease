import logging
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator
from app.schemas import HeartDataInput, PredictionOutput

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mlops_api")

app = FastAPI(
    title="Heart Disease Risk Prediction API",
    version="1.0.0"
)

# Instrument Prometheus metrics endpoint at /metrics
Instrumentator().instrument(app).expose(app)

MODEL_PATH = "models/best_model.pkl"

try:
    model = joblib.load(MODEL_PATH)
    logger.info("Model pipeline loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load model from {MODEL_PATH}: {e}")
    model = None

@app.get("/health")
def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Model unavailable")
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictionOutput)
def predict(data: HeartDataInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model unavailable")
    
    try:
        # Pydantic v2 compliant model export
        input_df = pd.DataFrame([data.model_dump()])
        prediction = int(model.predict(input_df)[0])
        probability = float(model.predict_proba(input_df)[0][1])
        
        risk_level = "High" if prediction == 1 else "Low"
        logger.info(f"Prediction: {prediction}, Probability: {probability:.4f}")
        
        return PredictionOutput(
            prediction=prediction,
            probability=probability,
            risk_level=risk_level
        )
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail="Inference failure")
