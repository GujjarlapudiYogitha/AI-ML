"""Load the ticket classifier and make predictions on new support tickets."""

from pathlib import Path

import joblib


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "calibrated_svm_pipeline.joblib"
MAX_TICKET_LENGTH = 10_000


def load_model(model_path=MODEL_PATH):
    """Load a saved ticket classification model."""
    model_path = Path(model_path)

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            "Run the Day 8 training notebook to generate it."
        )

    return joblib.load(model_path)

def validate_ticket(ticket):
    """Validate ticket type, content, and length."""
    if not isinstance(ticket, str):
        raise TypeError("Ticket must be a string.")

    cleaned_ticket = ticket.strip()

    if not cleaned_ticket:
        raise ValueError("Ticket must not be empty or whitespace-only.")

    if len(cleaned_ticket) > MAX_TICKET_LENGTH:
        raise ValueError(
            f"Ticket must not exceed {MAX_TICKET_LENGTH} characters."
        )

    return cleaned_ticket

def predict_ticket(ticket, model, confidence_threshold=0.60):
    """Predict a ticket category and recommend whether review is needed."""
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("Confidence threshold must be between 0 and 1.")

    cleaned_ticket = validate_ticket(ticket)

    probabilities = model.predict_proba([cleaned_ticket])[0]
    best_index = probabilities.argmax()

    category = str(model.classes_[best_index])
    confidence = float(probabilities[best_index])

    if confidence < confidence_threshold:
        decision = "Manual review"
    else:
        decision = "Eligible for automatic routing"

    return {
        "category": category,
        "confidence": confidence,
        "decision": decision
    }

if __name__ == "__main__":
    model = load_model()
    print("Model loaded successfully.")

    # test_inputs = [
    #     "  Payment failed  ",
    #     "",
    #     "   ",
    #     None,
    #     123
    # ]

    # for ticket in test_inputs:
    #     try:
    #         cleaned_ticket = validate_ticket(ticket)
    #         print(f"Accepted: {cleaned_ticket!r}")
    #     except (TypeError, ValueError) as error:
    #         print(f"Rejected {ticket!r}: {error}")

    # result = predict_ticket(
    #     "My credit card was charged twice. Please refund the duplicate payment.",
    #     model
    # )

    # print("Predicted category:", result["category"])
    # print(f"Confidence: {result['confidence']:.2%}")
    # print("Decision:", result["decision"])

    # review_result = predict_ticket(
    #     "My credit card was charged twice. Please refund the duplicate payment.",
    #     model,
    #     confidence_threshold=0.90
    # )

    # print("\nTesting the manual-review branch:")
    # print("Predicted category:", review_result["category"])
    # print(f"Confidence: {review_result['confidence']:.2%}")
    # print("Decision:", review_result["decision"])


    sample_ticket = (
        "My credit card was charged twice. "
        "Please refund the duplicate payment."
    )
    
    sample_result = predict_ticket(sample_ticket, model)

    boundary_result = predict_ticket(
        sample_ticket,
        model,
        confidence_threshold=sample_result["confidence"]
    )

    assert boundary_result["decision"] == "Eligible for automatic routing"
    print("\nExact-threshold check passed.")

    for invalid_threshold in [-0.1, 1.1]:
        try:
            predict_ticket(
                sample_ticket,
                model,
                confidence_threshold=invalid_threshold
            )
        except ValueError as error:
            print(f"Rejected threshold {invalid_threshold}: {error}")
        else:
            raise AssertionError("Invalid threshold was accepted.")