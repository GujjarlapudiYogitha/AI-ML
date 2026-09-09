# Customer Support Ticket Intelligence

## Problem Statement

Customer-support teams receive large numbers of tickets describing billing
questions, login problems, technical issues, and requests for product
improvements. Reading every ticket and manually assigning it to the correct
team takes time, delays the first response, and can result in inconsistent
routing.

This project will build a machine-learning application that analyzes the text
of a customer-support ticket and predicts its primary category. The
application will also display the model's confidence and recommend the support
team that should review the ticket. The goal is to demonstrate an end-to-end
text-classification workflow while keeping a human involved when a prediction
is uncertain.

## Project Goal

Given a non-empty customer-support message, the first version will:

1. Predict one of four ticket categories.
2. Display a confidence score for the prediction.
3. Recommend the appropriate support team.
4. Return uncertain or ambiguous tickets for human review rather than making
   an automatic final decision.

## Ticket Categories

| Category | Description | Suggested team |
| --- | --- | --- |
| Billing | Payments, duplicate charges, invoices, subscriptions, and refunds | Billing Support |
| Account Access | Login, password reset, locked account, and authentication problems | Account Support |
| Technical Support | Errors, crashes, unavailable functionality, and other product failures | Technical Support |
| Product Request | Requests for new features or improvements to existing functionality | Product Team |

Each ticket will receive one primary category. When a ticket mentions multiple
issues, the category representing the user's immediate support need should be
used as its label.

## Example

**Input**

```text
I was charged twice for my monthly subscription.
```

**Illustrative output**

```json
{
  "category": "Billing",
  "confidence": 0.87,
  "suggested_team": "Billing Support"
}
```

The confidence shown above is only an example. Measured scores will be added
after the model has been trained and evaluated.

## Version 1 Scope

Version 1 is a focused multiclass text-classification application. It will
include:

- Loading and validating a legally reusable support-ticket dataset
- Exploring ticket text, labels, missing values, duplicates, and class balance
- Mapping source labels into four documented categories
- Creating reproducible, stratified training and test splits
- Training a TF-IDF and Logistic Regression baseline pipeline
- Evaluating the model with macro precision, recall, F1, per-class results, and
  a confusion matrix
- Reviewing incorrect predictions and documenting important failure cases
- Saving the complete trained pipeline for repeatable inference
- Validating empty, invalid, and unusually long inputs
- Providing a Gradio interface for ticket classification
- Publishing the model documentation and interactive demo on Hugging Face
- Adding basic automated tests and reproducible setup instructions

## Success Criteria

The first version will be considered complete when:

- A user can submit a non-empty ticket through the interface.
- The application returns one of the four approved categories.
- The application displays prediction confidence and a suggested team.
- Model performance is reported using macro F1 and per-category metrics.
- Known limitations and ambiguous examples are documented.
- Automated tests pass in a clean local environment.
- The GitHub repository and Hugging Face demo link to each other.

No minimum model score is promised before the dataset is selected and a
baseline is measured.

## Not Included in Version 1

The following features are intentionally postponed to keep the first version
small enough to complete and understand:

- Ticket summarization
- Priority prediction
- Automatic email sending or ticket assignment
- Zendesk, Salesforce, or other external service integrations
- Database storage
- LLM or autonomous-agent workflows
- DistilBERT fine-tuning
- User authentication
- Production monitoring infrastructure

## Planned Technology Stack

- Python
- Pandas
- Scikit-learn
- Joblib
- Pytest
- Gradio
- Hugging Face Spaces