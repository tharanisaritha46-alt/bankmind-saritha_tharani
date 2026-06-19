# Project Process

This file explains the exact process used to build the BankMind Track C
submission.

## 1. Understand the Task

The challenge asks for one selected track. I selected **Track C - System
Builder**, which also includes the core Track B machine-learning requirements.

Track C requires:

- focused EDA
- a Logistic Regression baseline model
- a stronger main model
- accuracy, precision, recall, F1, and `classification_report`
- 5 readable sample predictions from the test set
- a saved model file
- a FastAPI app with `POST /predict` and `GET /health`
- a README with curl examples
- `EXPLANATION.md`
- optional bonus: `POST /explain` using Groq

## 2. Load and Inspect the Data

The project uses `data/bank-full.csv` from the UCI Bank Marketing dataset. The
file is semicolon-delimited, so it is loaded with:

```python
pd.read_csv(DATA_PATH, sep=";")
```

The script prints:

- dataset shape
- missing values
- target class distribution

The important finding is that only **11.70%** of customers subscribed. That
class imbalance affects the model evaluation strategy.

## 3. Preprocess Features

The preprocessing code lives in `bankmind/preprocessing.py` so training and API
serving use the same transformations.

Numeric columns are used directly:

- `age`
- `balance`
- `day`
- `duration`
- `campaign`
- `pdays`
- `previous`

Categorical columns are encoded with stable integer mappings:

- `job`
- `marital`
- `education`
- `default`
- `housing`
- `loan`
- `contact`
- `month`
- `poutcome`

The API can receive partial payloads. Missing fields are filled with medians for
numeric columns and modes for categorical columns.

## 4. Train Two Models

The baseline model is Logistic Regression with class weighting:

```python
LogisticRegression(max_iter=1000, class_weight="balanced")
```

The main model is Random Forest:

```python
RandomForestClassifier(
    n_estimators=150,
    max_depth=20,
    min_samples_leaf=4,
    class_weight={0: 1, 1: 3},
    random_state=42,
    n_jobs=-1,
)
```

The train/test split is stratified so the positive/negative class ratio is
preserved in both sets.

## 5. Evaluate the Models

The script reports:

- accuracy
- precision
- recall
- F1
- `classification_report`

F1 is treated as more important than raw accuracy because the dataset is
imbalanced. A model can look good on accuracy while missing most subscribers.

## 6. Interpret the Model

The training script prints Random Forest feature importances. The strongest
feature is `duration`.

This is explained carefully in `EXPLANATION.md`: `duration` is predictive, but
it is also target leakage because call duration is only known after the call.
For a production pre-call targeting system, the model should be retrained
without `duration`.

## 7. Save the Model Artifact

The saved file is `model/model.pkl`. It contains:

- trained Random Forest model
- category maps
- default values
- feature importances
- metrics
- yes-rate

Saving all of this together makes the API reproducible because it serves the
same feature encoding used during training.

## 8. Build the API

The FastAPI app is in `api/app.py`.

Required endpoint:

```text
GET /health
```

It returns API/model status.

Required endpoint:

```text
POST /predict
```

It returns:

- `will_subscribe`
- `probability`
- `top_factors`

Bonus endpoint:

```text
POST /explain
```

It uses Groq if `GROQ_API_KEY` is configured. If the key is missing or the Groq
call fails, the endpoint returns a deterministic fallback explanation instead of
crashing.

## 9. Add Review and Deployment Support

The project includes:

- `smoke_test.py` for quick local API verification
- `.github/workflows/ci.yml` for GitHub Actions smoke testing
- `render.yaml` for optional Render deployment
- `.env.example` for optional Groq configuration

## 10. How I Would Submit

I would submit the repository as:

```text
bankmind-saritha
```

Then I would:

1. Push the project to a public GitHub repository.
2. Confirm GitHub Actions passes.
3. Optionally deploy to Render using `render.yaml`.
4. Add the live API URL to the README if deployment succeeds.
5. Submit the GitHub repository link through the challenge form.
