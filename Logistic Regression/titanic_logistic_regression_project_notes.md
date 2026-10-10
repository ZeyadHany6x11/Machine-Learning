# Titanic Survival Prediction — Logistic Regression

## Project Overview

This project uses Seaborn's Titanic dataset to predict whether a passenger survived the Titanic disaster.

- **Target:** `survived`
  - `0` = did not survive
  - `1` = survived
- **Model:** Logistic Regression
- **Task type:** Binary classification
- **Notebook:** `titanic_logistic_regression_complete.ipynb`

The notebook includes explanatory Markdown cells before the code cells, so each stage is documented before it runs.

## What We Did

### 1. Exploratory Data Analysis (EDA)

Loaded the Titanic dataset with Seaborn and explored:

- First rows and dataset dimensions
- Column names and data types
- Non-null counts and missing-value counts/percentages
- Descriptive statistics for numeric and categorical columns
- Survivor and non-survivor counts and percentages
- Survival rates by sex and passenger class
- Exact duplicate rows, without automatically deleting them

### 2. Target and Feature Selection

Set `survived` as the target `y`.

Started with these features:

- `pclass`
- `sex`
- `age`
- `sibsp`
- `parch`
- `fare`
- `embarked`

We excluded `alive` because it directly reveals the target. We also avoided redundant alternatives such as keeping both `pclass` and `class`, or both `embarked` and `embark_town`. The high-missingness `deck` feature and identifier/free-text columns were left out of this first model.

### 3. Feature Engineering

Created two additional features:

- `family_size` = `sibsp` + `parch` + 1
- `is_alone` = 1 when `family_size` is 1, otherwise 0

These features are derived from existing passenger information and do not use the target.

### 4. Numerical and Categorical Features

Numerical features:

- `age`
- `sibsp`
- `parch`
- `fare`
- `family_size`
- `is_alone`

Categorical features:

- `pclass`
- `sex`
- `embarked`

Although `pclass` is represented by numbers, it is treated as categorical in this experiment because it represents ticket classes.

### 5. Train/Test Split

Split the data into:

- 80% training data
- 20% test data

Used `random_state=42` for reproducibility and `stratify=y` to keep the survival proportions approximately consistent between the sets.

**Important:** The split happens before fitting imputers, encoders, or scalers.

### 6. Preprocessing Pipeline

Built a scikit-learn `ColumnTransformer` with separate preprocessing pipelines.

**Numerical pipeline**
1. `SimpleImputer(strategy="median")`
2. `StandardScaler()`

**Categorical pipeline**
1. `SimpleImputer(strategy="most_frequent")`
2. `OneHotEncoder(handle_unknown="ignore")`

The preprocessing steps are combined with Logistic Regression in a `Pipeline`. This ensures that preprocessing is fitted on the training data only and then applied consistently to the test data, reducing the risk of data leakage.

### 7. Model Training

Trained a Logistic Regression model with `max_iter=2000` and a fixed random state.

Generated:
- Predicted class labels with `predict()`
- Predicted survival probabilities with `predict_proba()`

The notebook also displays the number of solver iterations used.

### 8. Model Evaluation

Evaluated predictions on the held-out test set using:

- Accuracy
- Precision
- Recall
- F1-score
- Classification report
- Confusion matrix

Compared training accuracy with test accuracy as a basic diagnostic for possible overfitting or underfitting.

The notebook also plots predicted survival probabilities and demonstrates how changing the classification threshold from 0.5 affects metrics. The test set should not be repeatedly used to choose a threshold.

### 9. Preprocessing Strategy Comparison

Compared two preprocessing configurations:

- Median imputation for numeric columns and most-frequent imputation for categorical columns
- Mean imputation for numeric columns and a dedicated missing category for categorical columns

Both alternatives are fitted using the training set, then compared using test metrics.

### 10. Hyperparameter Tuning

Used `GridSearchCV` with 5-fold stratified cross-validation on the training data to compare these `C` values:

- `0.01`
- `0.1`
- `1.0`
- `10.0`
- `100.0`

`C` is the inverse of regularization strength:
- Smaller `C` means stronger regularization.
- Larger `C` means weaker regularization.

The best configuration is selected through cross-validation, then evaluated on the held-out test set.

### 11. Coefficient Interpretation

Extracted the fitted model's feature names and Logistic Regression coefficients.

- Positive coefficients increase the model's log-odds of survival, holding other encoded features constant.
- Negative coefficients decrease the model's log-odds of survival.
- One-hot encoded category coefficients are interpreted relative to the omitted reference category.

These coefficients describe model associations and should not be interpreted as proof of causation.

### 12. Final Comparison and Conclusion

Created a final comparison table for the baseline and tuned models using the same test set.

The notebook ends with a conclusion checklist asking for:
1. Best model and reason
2. Features that appear influential according to coefficients
3. Main error type (false positives or false negatives)
4. Evidence of overfitting or underfitting
5. A limitation or next experiment

Fill in this conclusion using the actual outputs from your run; scores are not hard-coded because they can vary with library versions and execution environments.

## How to Run

1. Open `titanic_logistic_regression_complete.ipynb` in Jupyter Notebook or VS Code.
2. Select the Python environment where the required packages are installed.
3. Run the cells in order from top to bottom.
4. If a package is missing, install it in the same environment used by the notebook.
5. The first call to `sns.load_dataset("titanic")` may require internet access.

Typical dependencies used by the notebook include:

- `numpy`
- `pandas`
- `seaborn`
- `matplotlib`
- `scikit-learn`

## Key Lessons

- Explore missing data before deciding how to handle it.
- Do not automatically delete duplicate-looking feature rows; different passengers can share the same recorded features.
- Do not include target-revealing columns such as `alive` in `X`.
- Split data before fitting preprocessing transformations.
- Use pipelines to keep imputation, encoding, scaling, and modeling consistent.
- Use cross-validation on training data for model selection.
- Reserve the test set for final evaluation.
- Interpret scores and coefficients in context rather than treating them as proof of causation.

## Project Status

The notebook contains the complete workflow and code for the experiment. Actual model scores and conclusions should be filled in after running the notebook in your own environment.
