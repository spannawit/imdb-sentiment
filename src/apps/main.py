"""
main.py

FastAPI application for serving sentiment analysis predictions using a fine-tuned BERT model.
Includes:
- Swagger UI and ReDoc documentation
- CORS middleware for cross-origin support
- Health check endpoint
- Sentiment prediction endpoint
"""


from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import torch
from apps.model_utils import load_model, predict

app = FastAPI(
    title="IMDb Sentiment Analysis API",
    description="REST API for predicting movie review sentiment using a fine-tuned BERT model.",
    version="1.0.0"
)

# Enable CORS for Swagger UI or external clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model on startup
model, tokenizer = load_model("models/bert_finetuned")

class ReviewInput(BaseModel):
    """Request model containing a movie review string."""
    review: str

class PredictionOutput(BaseModel):
    """Response model containing predicted sentiment and confidence score."""
    label: str
    confidence: float

@app.get("/", summary="Health Check", response_description="Service status")
def read_root() -> dict[str, str]:
    """Health check endpoint to verify service is running."""
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionOutput, summary="Predict Sentiment", response_description="Predicted sentiment and confidence")
def predict_sentiment(input: ReviewInput) -> dict[str, float | str]:
    """
    Predict sentiment from a given movie review.

    Args:
        input (ReviewInput): Input review text.

    Returns:
        dict: Sentiment label and confidence score.
    """
    try:
        label, confidence = predict(model, tokenizer, input.review)
        return {"label": label, "confidence": confidence}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
