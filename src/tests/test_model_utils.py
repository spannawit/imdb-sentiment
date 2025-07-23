import pytest
from apps import model_utils

@pytest.fixture(scope="module")
def model_and_tokenizer():
    model_path = "models/bert_finetuned"
    model, tokenizer = model_utils.load_model(model_path)
    return model, tokenizer

def test_load_model(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    assert model is not None
    assert tokenizer is not None

def test_predict_valid(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    text = "The movie was a masterpiece, truly inspiring."
    label, confidence = model_utils.predict(model, tokenizer, text)
    assert label in model_utils.LABELS
    assert 0.0 <= confidence <= 1.0

def test_predict_empty(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    text = ""
    label, confidence = model_utils.predict(model, tokenizer, text)
    assert label in model_utils.LABELS
    assert 0.0 <= confidence <= 1.0
