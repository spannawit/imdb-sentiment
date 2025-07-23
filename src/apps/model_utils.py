"""
model_utils.py

Utility functions for loading and using a sentiment classification model.
Includes:
- Loading a DistilBERT-based classification model
- Predicting sentiment from raw text input
"""

import torch
from torch.nn.functional import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification

LABELS = ["negative", "positive"]

def load_model(model_path: str):
    """
    Load a pre-trained transformer model and tokenizer for sentiment classification.

    Args:
        model_path (str): Path to the directory containing the trained model.

    Returns:
        model (PreTrainedModel): Loaded transformer model.
        tokenizer (PreTrainedTokenizer): Corresponding tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return model, tokenizer

def predict(model, tokenizer, text: str) -> tuple[str, float]:
    """
    Predict sentiment of a given text using the model and tokenizer.

    Args:
        model (PreTrainedModel): Trained sentiment classification model.
        tokenizer (PreTrainedTokenizer): Tokenizer for preprocessing text.
        text (str): Input review text to classify.

    Returns:
        Tuple[str, float]: Sentiment label ('positive' or 'negative') and confidence score.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = softmax(outputs.logits, dim=1)
        confidence, predicted = torch.max(probs, dim=1)
        return LABELS[predicted.item()], round(confidence.item(), 4)
