import pytest
from src.preprocess import preprocess_amharic

def test_preprocess_amharic():
    text = "ዋጋ 1000 ብር በ Addis Abeba 😊"
    expected = "ዋጋ 1000 ETB በ Addis Abeba"
    assert preprocess_amharic(text) == expected

def test_preprocess_empty():
    assert preprocess_amharic("") == ""
    assert preprocess_amharic(None) == ""