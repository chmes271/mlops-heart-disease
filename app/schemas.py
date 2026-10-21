from pydantic import BaseModel, Field

class HeartDataInput(BaseModel):
    age: float = Field(..., json_schema_extra={"example": 57.0})
    sex: float = Field(..., json_schema_extra={"example": 1.0})
    cp: float = Field(..., json_schema_extra={"example": 4.0})
    trestbps: float = Field(..., json_schema_extra={"example": 140.0})
    chol: float = Field(..., json_schema_extra={"example": 241.0})
    fbs: float = Field(..., json_schema_extra={"example": 0.0})
    restecg: float = Field(..., json_schema_extra={"example": 0.0})
    thalach: float = Field(..., json_schema_extra={"example": 123.0})
    exang: float = Field(..., json_schema_extra={"example": 1.0})
    oldpeak: float = Field(..., json_schema_extra={"example": 0.2})
    slope: float = Field(..., json_schema_extra={"example": 2.0})
    ca: float = Field(..., json_schema_extra={"example": 0.0})
    thal: float = Field(..., json_schema_extra={"example": 7.0})

class PredictionOutput(BaseModel):
    prediction: int
    probability: float
    risk_level: str
