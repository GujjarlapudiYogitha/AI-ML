import pytest
from src.predict import (
    MAX_TICKET_LENGTH,
    load_model,
    predict_ticket,
    validate_ticket
)
import numpy as np

def test_validate_ticket_strips_surrounding_whitespace():
    ticket = "  Payment failed \n"

    cleaned_ticket = validate_ticket(ticket)

    assert cleaned_ticket == "Payment failed"

def test_validate_ticket_rejects_empty_text():
    with pytest.raises(ValueError, match="must not be empty"):
        validate_ticket("")


def test_validate_ticket_rejects_non_text():
    with pytest.raises(TypeError, match="must be a string"):
        validate_ticket(None)

def test_validate_ticket_rejects_whitespace_only():
    with pytest.raises(ValueError, match="must not be empty"):
        validate_ticket(" \t\n ")

def test_validate_ticket_accepts_maximum_length():
    ticket = "a" * MAX_TICKET_LENGTH

    assert validate_ticket(ticket) == ticket


def test_validate_ticket_rejects_over_maximum_length():
    ticket = "a" * (MAX_TICKET_LENGTH + 1)

    with pytest.raises(ValueError, match="must not exceed"):
        validate_ticket(ticket)

def test_load_model_rejects_missing_file(tmp_path):
    missing_model_path = tmp_path / "missing_model.joblib"

    with pytest.raises(FileNotFoundError, match="Model file not found"):
        load_model(missing_model_path)


class FixedProbabilityModel:
    classes_ = np.array([
        "Billing and Payments",
        "Customer Service",
        "Product Support",
        "Technical Support"
    ])

    def predict_proba(self, tickets):
        return np.array([[0.80, 0.10, 0.05, 0.05] for _ in tickets])


def test_predict_ticket_allows_high_confidence_routing():
    model = FixedProbabilityModel()

    result = predict_ticket("Payment failed", model)

    assert result["category"] == "Billing and Payments"
    assert result["confidence"] == pytest.approx(0.80)
    assert result["decision"] == "Eligible for automatic routing"

def test_predict_ticket_requires_review_below_threshold():
    model = FixedProbabilityModel()

    result = predict_ticket(
        "Payment failed",
        model,
        confidence_threshold=0.90
    )

    assert result["decision"] == "Manual review"


def test_predict_ticket_allows_routing_at_exact_threshold():
    model = FixedProbabilityModel()

    result = predict_ticket(
        "Payment failed",
        model,
        confidence_threshold=0.80
    )

    assert result["decision"] == "Eligible for automatic routing"


@pytest.mark.parametrize("threshold", [-0.1, 1.1])
def test_predict_ticket_rejects_invalid_threshold(threshold):
    model = FixedProbabilityModel()

    with pytest.raises(
        ValueError,
        match="Confidence threshold must be between 0 and 1"
    ):
        predict_ticket(
            "Payment failed",
            model,
            confidence_threshold=threshold
        )