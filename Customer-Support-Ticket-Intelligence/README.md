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
| Billing and Payments | Payments, charges, invoices, subscriptions, and other billing questions | Billing and Payments |
| Customer Service | General customer questions and service-related requests | Customer Service |
| Product Support | Questions and issues involving a product or its functionality | Product Support |
| Technical Support | Errors, crashes, unavailable functionality, and other product failures | Technical Support |

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
  "category": "Billing and Payments",
  "confidence": 0.87,
  "suggested_team": "Billing and Payments"
}
```

The confidence shown above is only an example. Measured scores will be added
after the model has been trained and evaluated.

## Dataset

This project uses the
[Customer Support Tickets dataset](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets)
published by Tobi-Bueck on Hugging Face. The dataset contains synthetic
customer-support emails and is licensed under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). It is used
here for a noncommercial educational portfolio project with attribution to the
dataset creator.

The downloaded source file contains 28,587 records. Version 1 filters the data
to 11,815 English-language tickets assigned to four queues: Billing and
Payments, Customer Service, Product Support, and Technical Support.

The planned model input combines the ticket `subject` and `body`, while the
prediction target is `queue`. The `answer` column is excluded because it is
written after a ticket is handled and could reveal information about the
correct queue, causing data leakage.

Within the filtered data, 1,842 records have a missing subject, while `body`
and `queue` have no missing values. A missing subject can therefore be replaced
with an empty string while retaining the ticket body. The four classes are
moderately imbalanced, with Technical Support representing the largest class
and Billing and Payments the smallest. Macro F1 and per-class metrics will be
used so that performance on smaller classes is not hidden by overall accuracy.

Manual inspection also found some ambiguous or potentially noisy queue labels.
This label quality, along with the synthetic nature of the tickets, is a known
limitation and will be considered during error analysis.

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
