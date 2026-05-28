# 💳 Credit Card Fraud Detection ML

An end-to-end machine learning project for detecting fraudulent credit card
transactions using the Kaggle Credit Card Fraud Detection dataset.

This project includes a full Python training pipeline, a Streamlit dashboard,
model comparison reports, prediction tools, feature importance analysis, and
original Jupyter notebooks.

---

## ✨ Features

- 🧠 Train multiple ML models
- 📊 Compare model accuracy, precision, recall, and F1-score
- 🚨 Focus on fraud-class performance, not only accuracy
- 📁 Auto-detect the best available dataset
- 📉 Generate confusion matrix charts
- 🔍 Generate dataset quality reports
- 🧾 Save dataset profile metadata
- 🧮 Export feature importance reports
- 🎚️ Adjust fraud probability threshold
- 🧪 Score a single manual transaction from the UI
- 📤 Upload CSV files for batch prediction
- 📥 Download prediction results
- 💾 Save the best trained model
- 📓 Keep original notebook experiments available

---

## 📌 Current Dataset

The project auto-detects datasets in this order:

```text
data/raw/kaggel-creditcard.csv
data/raw/kaggle-creditcard.csv
data/raw/creditcard.csv
```

Your real Kaggle dataset is:

```text
data/raw/kaggel-creditcard.csv
```

The smaller file below is synthetic demo data:

```text
data/raw/creditcard.csv
```

The pipeline and UI prefer the real Kaggle file when it exists.

---

## 🧾 Dataset Format

Expected columns:

- `Time`
- `V1` to `V28`
- `Amount`
- `Class`

Target values:

- `0` = normal transaction
- `1` = fraudulent transaction

The Kaggle dataset is highly imbalanced. In the real dataset, fraud transactions
are a very small percentage of all transactions, so recall and F1-score are more
important than accuracy alone.

---

## 🗂️ Project Structure

```text
.
|-- app/
|   `-- streamlit_app.py              # Streamlit dashboard
|-- data/
|   |-- raw/
|   |   |-- kaggel-creditcard.csv     # Real Kaggle dataset
|   |   `-- creditcard.csv            # Synthetic demo dataset
|   `-- processed/
|-- models/
|   `-- best_model.joblib             # Saved best trained model
|-- notebooks/
|   |-- fraud_detection.ipynb
|   |-- model_comparison.ipynb
|   |-- logistic_regression.ipynb
|   |-- decision_tree.ipynb
|   |-- random_forest.ipynb
|   `-- knn.ipynb
|-- reports/
|   |-- figures/                      # Generated charts
|   |-- model_results_summary.csv
|   |-- training_metadata.json
|   |-- dataset_profile.json
|   |-- data_quality_report.csv
|   `-- feature_importance.csv
|-- src/
|   |-- create_sample_dataset.py
|   |-- fraud_detection_pipeline.py
|   `-- predict.py
|-- requirements.txt
`-- README.md
```

---

## ⚙️ Setup

Open PowerShell in the project folder:

```powershell
cd "C:\Users\USER\Downloads\credit-card-fraud-detection-ml-main\credit-card-fraud-detection-ml-main"
```

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

---

## 🚀 Run The Streamlit UI

Start the app:

```powershell
python -m streamlit run app/streamlit_app.py
```

Open:

```text
http://localhost:8501
```

### 🖥️ UI Tabs

**Train**

- Shows active dataset
- Shows row/class counts
- Trains all models
- Saves model reports

**Insights**

- Shows fraud rate
- Shows missing values
- Shows amount statistics
- Shows data quality report
- Shows feature importance

**Reports**

- Shows best model
- Shows model comparison table
- Shows confusion matrices
- Shows feature importance charts
- Shows training metadata source

**Predict**

- Upload CSV for batch scoring
- Adjust fraud threshold
- Download prediction output
- Manually score one transaction

---

## 🧠 Train Models From CLI

Run training using the auto-detected dataset:

```powershell
python src/fraud_detection_pipeline.py
```

Run training with a specific dataset:

```powershell
python src/fraud_detection_pipeline.py --data data/raw/kaggel-creditcard.csv
```

Training creates:

```text
models/best_model.joblib
reports/model_results_summary.csv
reports/training_metadata.json
reports/dataset_profile.json
reports/data_quality_report.csv
reports/feature_importance.csv
reports/figures/
```

---

## 🤖 Models Used

The pipeline trains and compares:

- Logistic Regression
- Decision Tree
- Random Forest
- KNN

The best model is selected using fraud F1-score and fraud recall.

---

## 📊 Latest Real Dataset Result

The current run used:

```text
data/raw/kaggel-creditcard.csv
```

Latest training summary:

```text
Rows: 284,807
Normal transactions: 284,315
Fraud transactions: 492
Best model: Random Forest
Accuracy: 0.9324
Fraud precision: 0.9638
Fraud recall: 0.8986
Fraud F1-score: 0.9301
```

---

## 🔮 Run Batch Prediction

After training:

```powershell
python src/predict.py data/raw/kaggel-creditcard.csv --output reports/predictions.csv --threshold 0.50
```

Output columns added:

- `Predicted_Class`
- `Fraud_Probability`, when supported by the model

Threshold meaning:

- Lower threshold = catches more possible fraud, but may create more false alerts
- Higher threshold = stricter fraud prediction, but may miss more fraud

---

## 📈 Generated Reports

Main report files:

```text
reports/model_results_summary.csv
reports/training_metadata.json
reports/dataset_profile.json
reports/data_quality_report.csv
reports/feature_importance.csv
```

Generated charts:

```text
reports/figures/class_distribution.png
reports/figures/amount_by_class.png
reports/figures/*_confusion_matrix.png
reports/figures/*_feature_importance.png
```

---

## 🧩 Important Functions

Main project functions:

- `resolve_dataset_path()` auto-selects the best available dataset
- `load_dataset()` validates dataset schema
- `dataset_profile()` summarizes dataset size and fraud rate
- `create_data_quality_report()` exports missing values and numeric stats
- `balance_dataset()` undersamples normal transactions
- `train_and_evaluate()` trains and compares all models
- `extract_feature_importance()` exports model feature importance
- `save_eda_outputs()` creates charts and data reports
- `predict_transactions()` scores transactions with a threshold

---

## 🧪 Optional Demo Dataset

Only use this when you do not have the real Kaggle dataset:

```powershell
python src/create_sample_dataset.py
```

This creates:

```text
data/raw/creditcard.csv
```

⚠️ This file is synthetic. Do not use it for final results.

---

## 📓 Notebooks

The original notebooks are kept in:

```text
notebooks/
```

Recommended notebook:

```text
notebooks/model_comparison.ipynb
```

For the most complete and reliable workflow, use the Python pipeline or
Streamlit UI.

---

## 🛠️ Troubleshooting

### Streamlit app shows old code

Hard refresh the browser:

```text
Ctrl + F5
```

Restart Streamlit:

```powershell
python -m streamlit run app/streamlit_app.py
```

### Dataset not found

Make sure one of these files exists:

```text
data/raw/kaggel-creditcard.csv
data/raw/kaggle-creditcard.csv
data/raw/creditcard.csv
```

### Prediction CSV fails

Make sure the CSV includes:

```text
Time, V1, V2, ..., V28, Amount
```

`Class` is optional for prediction.

---

## ⚠️ Important ML Note

This project uses undersampling to balance the training data. That is useful for
learning and quick comparison, but real fraud systems should also evaluate on
the original imbalanced distribution and consider:

- class weights
- threshold tuning
- cost-sensitive evaluation
- false-positive business cost
- false-negative fraud cost

---

## ✅ Project Status

The project is runnable with:

```powershell
python -m streamlit run app/streamlit_app.py
```

or:

```powershell
python src/fraud_detection_pipeline.py
```
