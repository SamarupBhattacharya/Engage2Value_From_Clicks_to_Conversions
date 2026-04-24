# Engage2Value: From Clicks to Conversions

> **🏆 4th Place** out of 1,500+ participants — Kaggle Competition: *System Threat Prediction / Engage2Value*

## Overview

This repository contains the solution for the **Engage2Value** Kaggle competition, which challenges participants to predict a user's `purchaseValue` from multi-session behavioral data collected across digital touchpoints. The approach leverages careful feature engineering, multi-stage data cleaning, and a pre-trained Random Forest Regressor to achieve a top-4 finish.

### Problem Statement

Given anonymized session-level data from a large-scale digital commerce platform, the goal is to predict the **total purchase amount** (`purchaseValue`) for each user session. Features include browser types, traffic sources, device details, and geographical indicators.

---

## Repository Structure

```
.
├── engage2value.ipynb   # Full solution pipeline (EDA → preprocessing → prediction)
└── README.md
```

---

## Solution Pipeline

### 1. Data Loading

Training and test data are loaded from the Kaggle input directory:

```
/kaggle/input/engage-2-value-from-clicks-to-conversions/train_data.csv
/kaggle/input/engage-2-value-from-clicks-to-conversions/test_data.csv
```

The target variable `purchaseValue` is separated from the training features.

---

### 2. Automated Column Profiling

A custom `summarize_dataframe_columns()` utility generates a per-column summary covering:

| Field | Description |
|---|---|
| `Type` | Numeric or Categorical |
| `NaN_pcnt` | Percentage of missing values |
| `N_Unique` | Count of unique non-null values |

This summary drives all subsequent cleaning decisions.

---

### 3. Feature Cleaning (Multi-Step)

**Step 1 — Column Dropping**

Columns are removed if any of the following conditions hold:
- More than 90% of values are missing (`NaN_pcnt > 90`)
- No variance: zero missing values but only one unique value (`N_Unique == 1 & NaN_pcnt == 0`)
- Extremely high cardinality: more than 100,000 unique values (`N_Unique > 100,000`)

**Step 2 — Unknown Category Handling**

Categorical values present in the test set but absent from training are replaced with `NaN` to prevent data leakage and encoding errors at inference time.

**Step 3 — Special-Case Imputation**

For columns with exactly one unique non-null value and some missing entries:
- **Numeric:** missing values filled with `0`
- **Categorical:** binary-encoded as `1` (known value) or `0` (missing/other)

**Step 4 — Standard Imputation**

For columns with less than 1% missing values:
- **Numeric:** median imputation via `sklearn.SimpleImputer`
- **Categorical:** most-frequent (mode) imputation via `sklearn.SimpleImputer`

---

### 4. Date Feature Engineering

The `date` column (format `YYYYMMDD`) is decomposed into rich temporal features that capture seasonality and behavioral patterns:

| Feature | Description |
|---|---|
| `year`, `month`, `day` | Calendar components |
| `day_of_week` | 0 (Monday) – 6 (Sunday) |
| `day_of_year` | 1–366 |
| `week_of_year` | ISO calendar week |
| `quarter` | Q1–Q4 |
| `is_month_start` / `is_month_end` | Binary flags |
| `is_weekend` | Binary flag (Saturday or Sunday) |

The original `date` column is dropped after extraction.

---

### 5. KNN Imputation for High-Missingness Categorical Columns

Two high-cardinality categorical columns — `trafficSource.keyword` and `trafficSource.referralPath` — require special handling:

1. **Label Encoding** — categories mapped to integers while preserving `NaN` positions
2. **KNN Imputation** — a pre-fitted `KNNImputer` (loaded from a Kaggle dataset) fills missing values using neighborhood information from all numeric features
3. **Inverse Transform** — integer labels are decoded back to their original string categories

---

### 6. Encoding & Scaling

- **Numeric features:** standardized with `sklearn.StandardScaler` (fit on train, applied to test)
- **Categorical features:** one-hot encoded with `sklearn.OneHotEncoder` (fit on train, `handle_unknown='ignore'` for robustness on unseen test categories)

---

### 7. Prediction

A pre-trained `RandomForestRegressor` (loaded via `joblib` from a Kaggle dataset) generates predictions on the preprocessed test set. Predictions are saved to `submission.csv` in the required format:

```csv
id,purchaseValue
0,12.5
1,0.0
...
```

---

## Key Design Decisions

- **Pre-trained artifacts over in-notebook training** — both the `KNNImputer` and the `RandomForestRegressor` are loaded from saved `.joblib` files, ensuring reproducibility and avoiding re-training overhead on Kaggle's compute limits.
- **Train-only fitting** — all scalers, encoders, and imputers are fit exclusively on training data and applied to the test set, preventing data leakage.
- **Layered missing value strategy** — rather than a one-size-fits-all imputer, the pipeline applies targeted strategies based on the missingness pattern and column type of each feature.

---

## Dependencies

```
pandas
numpy
scikit-learn
joblib
```

All dependencies are available in the standard Kaggle Python environment.

---

## External Kaggle Datasets

The following pre-trained artifacts must be available as Kaggle input datasets:

| File | Usage |
|---|---|
| `knn_imputer.joblib` | KNN imputer for `trafficSource.*` columns |
| `random_forest_regressor_model.joblib` | Final prediction model |

Both are loaded from `/kaggle/input/knn-imputer/`.

---

## Results

| Metric | Value |
|---|---|
| Competition Rank | **4th / 1,500+** |
| Target Variable | `purchaseValue` (regression) |

---

## Usage

1. Add the notebook to a Kaggle competition environment with the `engage-2-value-from-clicks-to-conversions` dataset attached.
2. Attach the `knn-imputer` dataset containing the pre-trained `.joblib` files.
3. Run all cells in `engage2value.ipynb` sequentially.
4. The output `submission.csv` will be generated in the working directory.
