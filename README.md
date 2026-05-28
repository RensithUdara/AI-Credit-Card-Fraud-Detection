# Credit Card Fraud Detection ML

A complete machine learning project for credit card fraud detection using the
Kaggle Credit Card Fraud Detection dataset.

The project now includes:

- reusable Python training pipeline
- Streamlit UI dashboard
- model comparison reports
- saved best model
- batch prediction script
- original Jupyter notebooks
- optional synthetic sample data generator for testing only

## Active Dataset

The project auto-detects datasets in this order:

```text
data/raw/kaggel-creditcard.csv
data/raw/kaggle-creditcard.csv
data/raw/creditcard.csv
```

Your real Kaggle dataset is currently:

```text
data/raw/kaggel-creditcard.csv
```

The smaller `data/raw/creditcard.csv` file is synthetic demo data. The training
pipeline and UI now prefer the real Kaggle file when it exists.

Expected columns:

- `Time`
- `V1` to `V28`
- `Amount`
- `Class`

`Class` values:

- `0` = normal transaction
- `1` = fraudulent transaction

## Project Structure

```text
.
|-- app/
|   `-- streamlit_app.py
|-- data/
|   |-- raw/
|   |   |-- kaggel-creditcard.csv
|   |   `-- creditcard.csv
|   `-- processed/
|-- models/
|   `-- best_model.joblib
|-- notebooks/
|-- reports/
|   |-- figures/
|   |-- model_results_summary.csv
|   `-- training_metadata.json
|-- src/
|   |-- create_sample_dataset.py
|   |-- fraud_detection_pipeline.py
|   `-- predict.py
|-- requirements.txt
`-- README.md
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run Training

```powershell
python src/fraud_detection_pipeline.py
```

Optional custom dataset:

```powershell
python src/fraud_detection_pipeline.py --data data/raw/kaggel-creditcard.csv
```

Training creates:

```text
models/best_model.joblib
reports/model_results_summary.csv
reports/training_metadata.json
reports/figures/
```

## Run UI App

```powershell
python -m streamlit run app/streamlit_app.py
```

Then open:

```text
http://localhost:8501
```

The UI lets you:

- inspect the active dataset
- train all models
- view model metrics
- view confusion matrix charts
- upload a CSV for batch prediction
- download predictions

## Run Prediction

After training:

```powershell
python src/predict.py data/raw/kaggel-creditcard.csv --output reports/predictions.csv
```

Prediction output adds:

- `Predicted_Class`
- `Fraud_Probability`, if the selected model supports probabilities

## How The ML Works

1. Load the active Kaggle-format CSV.
2. Validate required columns.
3. Show original class distribution.
4. Balance the training data by undersampling normal transactions.
5. Split balanced data into train/test sets.
6. Scale features with `StandardScaler`.
7. Train:
   - Logistic Regression
   - Decision Tree
   - Random Forest
   - KNN
8. Compare models using:
   - Accuracy
   - Fraud precision
   - Fraud recall
   - Fraud F1-score
   - Confusion matrix values
9. Save the best model based on fraud F1-score and recall.

## Optional Demo Dataset

Only use this when you do not have the real Kaggle dataset:

```powershell
python src/create_sample_dataset.py
```

This creates `data/raw/creditcard.csv`, but it is synthetic and should not be
used for final results.

## Important Note

The current pipeline uses undersampling to handle the strong class imbalance.
This is good for a learning project and fast model comparison. For production
fraud detection, also evaluate on the original imbalanced distribution and
consider class weights, threshold tuning, and cost-sensitive metrics.
