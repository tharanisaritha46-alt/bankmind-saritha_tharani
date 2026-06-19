"""FastAPI service that serves the BankMind term-deposit model.

Endpoints:
  GET  /health   - sanity check
  POST /predict  - subscription prediction for a customer
  POST /explain  - plain-English explanation of a prediction (uses Groq if a key
                   is configured, otherwise a deterministic templated fallback)

Run:  uvicorn api.app:app --reload
"""
from __future__ import annotations

import os
from typing import Literal

import joblib
import requests
from fastapi import FastAPI
from pydantic import BaseModel, Field

from bankmind.preprocessing import MODEL_PATH, encode_frame, row_from_payload

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

app = FastAPI(
    title="BankMind API",
    description="Predicts which customers are likely to subscribe to a term deposit.",
    version="1.0.0",
)

# Loaded once at import time so every request reuses the same in-memory model.
if not MODEL_PATH.exists():
    raise RuntimeError(f"Model not found at {MODEL_PATH}. Run `python train.py` first.")
ARTIFACT = joblib.load(MODEL_PATH)
MODEL = ARTIFACT["model"]
CATEGORY_MAPS = ARTIFACT["category_maps"]
DEFAULTS = ARTIFACT["defaults"]
IMPORTANCES = ARTIFACT["feature_importances"]


class Customer(BaseModel):
    """Customer payload. Only a few fields are required; the rest fall back to
    dataset-derived defaults so callers can send partial records."""

    age: int = Field(45, ge=18, le=100)
    balance: float = Field(0, description="Average yearly balance in euros")
    housing: Literal["yes", "no"] = Field("no", description="Has a housing loan?")
    loan: Literal["yes", "no"] = Field("no", description="Has a personal loan?")
    job: str | None = None
    marital: str | None = None
    education: str | None = None
    default: str | None = None
    contact: str | None = None
    day: int | None = None
    month: str | None = None
    poutcome: str | None = None
    duration: int | None = Field(None, description="Last contact duration (seconds)")
    campaign: int | None = None
    pdays: int | None = None
    previous: int | None = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 64,
                "job": "retired",
                "balance": 7000,
                "housing": "no",
                "loan": "no",
                "duration": 600,
            }
        }
    }


class PredictResponse(BaseModel):
    will_subscribe: bool
    probability: float
    top_factors: list[str]


class ExplainResponse(BaseModel):
    will_subscribe: bool
    probability: float
    explanation: str
    source: str


def _predict(customer: Customer) -> tuple[float, list[str]]:
    payload = {k: v for k, v in customer.model_dump().items() if v is not None}
    frame = row_from_payload(payload, DEFAULTS)
    encoded = encode_frame(frame, CATEGORY_MAPS)
    proba = float(MODEL.predict_proba(encoded)[0, 1])
    return proba, _top_factors(payload)


def _top_factors(payload: dict) -> list[str]:
    """Human-readable drivers, ranked by global feature importance and gated on
    whether the customer's value is actually notable."""
    factors: list[tuple[float, str]] = []
    balance = float(payload.get("balance", DEFAULTS["balance"]))
    if balance >= 1500:
        factors.append((IMPORTANCES["balance"], "high account balance"))
    elif balance <= 0:
        factors.append((IMPORTANCES["balance"], "low/negative account balance"))

    if payload.get("housing", "no") == "no" and payload.get("loan", "no") == "no":
        factors.append((IMPORTANCES["housing"], "no existing loans"))
    elif payload.get("housing") == "yes" or payload.get("loan") == "yes":
        factors.append((IMPORTANCES["housing"], "has existing loan(s)"))

    duration = payload.get("duration")
    if duration is not None and duration >= 300:
        factors.append((IMPORTANCES["duration"], "long last-contact duration"))

    age = int(payload.get("age", DEFAULTS["age"]))
    if age >= 60:
        factors.append((IMPORTANCES["age"], "older customer (often retired)"))

    if payload.get("poutcome") == "success":
        factors.append((IMPORTANCES["poutcome"], "previous campaign succeeded"))

    factors.sort(reverse=True)
    ranked = [label for _, label in factors[:3]]
    return ranked or ["no strong distinguishing factors"]


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model": ARTIFACT["model_name"],
        "yes_rate": round(ARTIFACT.get("yes_rate", 0.117), 4),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(customer: Customer) -> PredictResponse:
    proba, factors = _predict(customer)
    return PredictResponse(
        will_subscribe=proba >= 0.5,
        probability=round(proba, 4),
        top_factors=factors,
    )


def _groq_explanation(customer: Customer, proba: float) -> str:
    prompt = (
        "Customer profile:\n"
        f"- Age: {customer.age}, Job: {customer.job or 'unknown'}, "
        f"Balance: {customer.balance}\n"
        f"- Existing loans: Housing={customer.housing}, Personal={customer.loan}\n"
        f"- Model prediction: {proba * 100:.0f}% chance of subscribing\n\n"
        "In 2-3 sentences, explain why this customer would or would not likely "
        "subscribe to a term deposit, and how an RM should approach the conversation."
    )
    resp = requests.post(
        GROQ_URL,
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        json={
            "model": GROQ_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.4,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def _template_explanation(customer: Customer, proba: float, factors: list[str]) -> str:
    lean = "likely" if proba >= 0.5 else "unlikely"
    drivers = ", ".join(factors)
    return (
        f"This customer is {lean} to subscribe ({proba * 100:.0f}% probability). "
        f"Key drivers: {drivers}. "
        "An RM should "
        + (
            "lead with the long-term savings benefits of a term deposit and move to close."
            if proba >= 0.5
            else "focus on building rapport and surfacing a need before pitching the product."
        )
    )


@app.post("/explain", response_model=ExplainResponse)
def explain(customer: Customer) -> ExplainResponse:
    proba, factors = _predict(customer)
    if GROQ_API_KEY:
        try:
            text = _groq_explanation(customer, proba)
            source = "groq"
        except Exception as exc:  # noqa: BLE001 - fall back, never 500 on LLM hiccups
            text = _template_explanation(customer, proba, factors)
            source = f"fallback (groq error: {exc})"
    else:
        text = _template_explanation(customer, proba, factors)
        source = "fallback (no GROQ_API_KEY set)"
    return ExplainResponse(
        will_subscribe=proba >= 0.5,
        probability=round(proba, 4),
        explanation=text,
        source=source,
    )
