# Data Folder

Place the Kaggle credit card fraud dataset here:

```text
data/raw/creditcard.csv
```

The original dataset is not committed because it is large and `.gitignore`
excludes CSV files.

Expected columns:

- `Time`
- `V1` to `V28`
- `Amount`
- `Class`

`Class` values:

- `0` = normal transaction
- `1` = fraudulent transaction
