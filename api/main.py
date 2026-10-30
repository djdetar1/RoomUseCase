"""FastAPI inference service. Each POST /predict scores one sensor reading.

    uvicorn api.main:app --reload
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from occupancy import __version__
from occupancy.predict import load_model, predict_one

app = FastAPI(
    title="Occupancy Detection API",
    version=__version__,
    description="Room occupancy prediction service.",
)


class SensorReading(BaseModel):
    date: str = Field(..., examples=["2015-02-04 17:51:00"])
    Temperature: float = Field(..., examples=[23.18])
    Humidity: float = Field(..., examples=[27.27])
    Light: float = Field(..., examples=[426.0])
    CO2: float = Field(..., examples=[721.25])
    HumidityRatio: float = Field(..., examples=[0.0047929])


class Prediction(BaseModel):
    occupancy: int = Field(..., description="1 = occupied, 0 = not occupied")
    probability: float = Field(..., description="P(occupied)")


@app.get("/health")
def health() -> dict:
    try:
        load_model()
        model_loaded = True
    except FileNotFoundError:
        model_loaded = False
    return {"status": "ok", "version": __version__, "model_loaded": model_loaded}


@app.post("/predict", response_model=Prediction)
def predict(reading: SensorReading) -> Prediction:
    try:
        result = predict_one(reading.model_dump())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return Prediction(**result)
