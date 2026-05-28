# Credit Card Fraud Detection ML

Machine learning project for detecting fraudulent credit card transactions using
the Kaggle credit card fraud dataset.

The project includes the original notebooks plus a clean Python pipeline that
can train, compare, save, and reuse fraud detection models.

## Project Structure

```text
.
├── data/
│   ├── raw/                  # Put creditcard.csv here
│   └── processed/
├── models/                   # Saved trained model
├── notebooks/                # Original Jupyter notebooks
├── reports/
│   ├── figures/              # Confusion matrices and class distribution plot
│   └── model_results_summary.csv
├── src/
│   ├── fraud_detection_pipeline.py
│   └── predict.py
├── requirements.txt
└── README.md
```

## Missing File

This repository can run with a small synthetic demo dataset, but real model
results require the real dataset.

Download the Kaggle dataset named **Credit Card Fraud Detection** and place the
CSV file here:

```text
data/raw/creditcard.csv
```

To create a small local demo CSV with the same columns:

```powershell
python src/create_sample_dataset.py
```

That creates:

```text
data/raw/creditcard.csv
```

The generated file is only for testing the code. Do not use it as real fraud
detection evidence or final project results.

Expected dataset columns:

- `Time`
- `V1` to `V28`
- `Amount`
- `Class`

`Class` means:

- `0` = normal transaction
- `1` = fraud transaction

## How It Works

1. Load `creditcard.csv`.
2. Validate the expected columns.
3. Check the original class distribution.
4. Balance the dataset by undersampling normal transactions.
5. Split data into training and testing sets.
6. Scale features using `StandardScaler`.
7. Train four models:
   - Logistic Regression
   - Decision Tree
   - Random Forest
   - KNN
8. Compare models using accuracy, precision, recall, and F1-score for fraud.
9. Save:
   - model summary CSV
   - confusion matrix images
   - best model as `models/best_model.joblib`

## Setup

From the project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run Training

```powershell
python src/create_sample_dataset.py
python src/fraud_detection_pipeline.py
```

Optional custom dataset path:

```powershell
python src/fraud_detection_pipeline.py --data path\to\creditcard.csv
```

After training, check:

```text
reports/model_results_summary.csv
reports/figures/
models/best_model.joblib
```

## Run Prediction

After training, predict fraud labels for a CSV file with the same feature
columns:

```powershell
python src/predict.py data/raw/creditcard.csv --output reports/predictions.csv
```

The prediction output adds:

- `Predicted_Class`
- `Fraud_Probability`, when the trained model supports probabilities

## Run Notebooks

Install dependencies, then start Jupyter:

```powershell
jupyter notebook
```

Recommended notebook:

```text
notebooks/model_comparison.ipynb
```

The notebooks were the original project work. The `src/` pipeline is the
cleaner and more reliable way to run the full project.

## Known Issues Fixed

- Added missing project folders: `data/`, `src/`, `models/`, and `reports/`.
- Added a runnable training pipeline.
- Added dataset validation and clear missing-file errors.
- Added model comparison output.
- Added saved plots and saved best model.
- Added prediction script.
- Added Jupyter and model-saving dependencies.

## Important Note

The project uses undersampling to balance fraud and normal classes. This makes
training faster and easier for learning, but in a real fraud detection system
you should also test other imbalance strategies and evaluate carefully on the
original imbalanced distribution.
