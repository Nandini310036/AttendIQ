# Advanced Student Attendance Analytics & Risk Prediction

An intermediate/advanced Python Data Science project featuring data cleaning, EDA, an interactive Streamlit dashboard, and machine-learning risk prediction.

## Important data note
The supplied student and attendance records are synthetic, created for learning and demonstration. Model scores on synthetic data are not evidence that the model will work on real students. Do not use predictions to penalize students.

## Features
- 200 sample students and 40 weekdays of attendance
- Data validation and duplicate handling
- Attendance summaries and interactive filters
- Attendance distribution, department comparison, and daily trend charts
- Logistic Regression and Random Forest comparison
- Risk prediction using early-period attendance features to predict a later-period outcome
- Precision, recall, F1-score, confusion matrix, downloadable CSVs

## Setup on Windows / VS Code
Open this project folder in VS Code and run in the terminal:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m src.train_model
streamlit run app.py
```

If PowerShell blocks virtual-environment activation, either run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and activate again, or skip activation and use `python -m pip install -r requirements.txt` followed by `python -m src.train_model` and `streamlit run app.py`.

## What happens
- `python -m src.train_model` creates a trained model in `models/`, model evaluation files and student predictions in `outputs/`.
- `streamlit run app.py` launches the interactive dashboard in your browser.
- You can train/refresh models from the Risk prediction tab as well.

## Project structure
- `data/`: input CSV files
- `src/data_processing.py`: loading, cleaning and feature engineering
- `src/train_model.py`: model training and evaluation
- `models/`: trained model and metadata (created after training)
- `outputs/`: cleaned data, metrics and predictions (created after training)
- `app.py`: interactive Streamlit dashboard

## Attendance formulas
Attendance percentage = present classes / total classes × 100.

The prediction task is defined as: use the first half of the dates to predict whether a student's attendance in the later half is below 75%. This split prevents using the later-period outcome as an input feature.

## Suggested improvements
- Replace sample data with approved, anonymized real records.
- Add subject-level attendance if data includes subject information.
- Use time-based validation and monitor performance on future terms.
- Add login and role-based access only if deploying with real data.
