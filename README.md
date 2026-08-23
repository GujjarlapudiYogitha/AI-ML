# Wind Turbine Drivetrain Fault Prediction

This project is a minimal data-science case study scaffold for the Aeolus Renewables drivetrain-condition problem described in the business context.

## Business context and objective

Aeolus Renewables operates a fleet of 1,150 onshore wind turbines across 16 wind farms. The business relies on consistent power generation under long-term power purchase agreements, so every hour a turbine is offline reduces saleable energy and revenue. The highest-risk components are the drivetrain, especially the gearbox and main bearing, which are expensive, slow to replace, and can cascade into larger failures if not treated early.

The objective is to build a machine-learning classifier that reads each turbine's sensor signature and labels the drivetrain as either `fault` or `normal`, with emphasis on catching genuine drivetrain degradation before a catastrophic failure occurs. This enables operation-centre analysts to focus on the small subset of turbines needing intervention instead of manually scanning the full fleet.

## Project workflow

1. Exploratory data analysis (EDA)
2. Data cleaning and preprocessing
3. Feature engineering
4. Baseline modelling and evaluation
5. Model interpretation and operational recommendations

## Dataset

The raw dataset is intentionally not committed because its redistribution
rights are not specified. Obtain `wind_turbine_detection.csv` from the course
or dataset provider and place it at:

```text
data/raw/wind_turbine_detection.csv
```

The empty `data/raw/` directory is retained so the expected location is clear.

## Project structure

```text
PROJECT/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── raw/
│       └── wind_turbine_detection.csv
├── AIML_Project_1_Full_Code_Notebook.ipynb
├── src/
│   ├── __init__.py
│   ├── eda.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   └── modeling.py
├── scripts/
│   ├── plot_failure.py
│   └── wind_speed_hist.py
└── .venv/ (optional local environment)
```

## Recommended workflow

- Use [src/eda.py](src/eda.py) to load the dataset and inspect distributions, missing values, and class balance.
- Use [src/preprocessing.py](src/preprocessing.py) to handle missing data, encode categorical variables, and split the data.
- Use [src/feature_engineering.py](src/feature_engineering.py) to add temporal and physics-informed features.
- Use [src/modeling.py](src/modeling.py) to train a simple baseline model and compare performance.

## Notes

- This repository contains the reproducible source and primary analysis notebook;
  generated plots, exports, and duplicate notebook templates are excluded.
- The target variable is the `failure` field, where the model aims to distinguish normal operation from fault conditions.
- Because missed drivetrain failures are more expensive than false alarms, model selection should prioritise recall for the fault class.
