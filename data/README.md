# 📁 Data Folder

This folder stores the datasets used by the credit card fraud detection project.

---

## ✅ Recommended Dataset

Place the real Kaggle Credit Card Fraud Detection CSV here:

```text
data/raw/kaggel-creditcard.csv
```

The current project is configured to use this file first.

---

## 🔎 Auto-Detection Order

The training pipeline and Streamlit UI automatically look for datasets in this
order:

```text
data/raw/kaggel-creditcard.csv
data/raw/kaggle-creditcard.csv
data/raw/creditcard.csv
```

If `kaggel-creditcard.csv` exists, it is used instead of the smaller demo file.

---

## 🧾 Expected Columns

The CSV must contain:

- `Time`
- `V1` to `V28`
- `Amount`
- `Class`

`Class` values:

- `0` = normal transaction
- `1` = fraudulent transaction

For prediction-only CSV files, `Class` is optional.

---

## 📌 Current Files

Typical files in this project:

```text
data/raw/kaggel-creditcard.csv   # Real Kaggle dataset
data/raw/creditcard.csv          # Synthetic demo dataset
data/processed/                  # Reserved for future processed data
```

---

## 🧪 Demo Dataset

If you do not have the real Kaggle dataset, create a small synthetic file:

```powershell
python src/create_sample_dataset.py
```

It creates:

```text
data/raw/creditcard.csv
```

⚠️ This is only for testing the app and code. It is not real fraud data.

---

## 🚫 Why CSV Files Are Ignored

The `.gitignore` contains:

```text
*.csv
```

Reason:

- real Kaggle data is large
- datasets should not usually be committed to GitHub
- generated prediction/report CSVs can change often

---

## 🧠 How The Dataset Is Used

The pipeline:

1. Loads the detected CSV
2. Validates required columns
3. Counts normal and fraud transactions
4. Creates dataset profile reports
5. Creates data quality reports
6. Balances the training set by undersampling normal transactions
7. Trains fraud detection models

Generated dataset reports are saved in:

```text
reports/dataset_profile.json
reports/data_quality_report.csv
```

---

## ⚠️ Important Note

The real Kaggle dataset is highly imbalanced. That is normal for fraud
detection. Fraud transactions are rare, so do not judge the model using accuracy
only. Look at:

- fraud recall
- fraud precision
- fraud F1-score
- missed fraud count
- false fraud alert count
