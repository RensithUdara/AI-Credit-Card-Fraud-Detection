# Data Folder

Place the Kaggle credit card fraud dataset in this folder.

```text
data/raw/kaggel-creditcard.csv
```

The pipeline auto-detects these names in order:

```text
data/raw/kaggel-creditcard.csv
data/raw/kaggle-creditcard.csv
data/raw/creditcard.csv
```

CSV files are ignored by `.gitignore` because the real dataset is large.

Expected columns:

- `Time`
- `V1` to `V28`
- `Amount`
- `Class`

`Class` values:

- `0` = normal transaction
- `1` = fraudulent transaction
