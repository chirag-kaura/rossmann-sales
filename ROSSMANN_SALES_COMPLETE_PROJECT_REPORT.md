# ROSSMANN SALES FORECASTING: COMPLETE TECHNICAL PROJECT REPORT

## SECTION 1 — EXECUTIVE SUMMARY
**Business Problem:** Rossmann, a European drugstore chain, needs to accurately forecast daily sales for over 1,000 stores up to six weeks in advance to optimize inventory management, staff scheduling, and promotional planning.
**Solution:** We developed a machine learning pipeline to predict daily store sales. We ingested raw historical sales data (over 1M records) and store metadata, cleaned it, and engineered robust features (temporal, rolling, lag, and competition metrics). 
**ML Approach:** We trained and evaluated tree-based regression models, specifically Random Forest and XGBoost. The final selected model was a Random Forest Regressor which achieved a Test RMSPE of 0.1199. 
**MLOps Implementation:** The project implements strict data versioning using DVC, experiment tracking using MLflow, and reproducible batch inference using a Docker container.
**Key Insights:** Promotions drive an 81% increase in sales. Sunday sales are almost non-existent because most stores are closed. Customer volume is highly correlated with sales (0.82) but is a potential leakage variable if used directly.

> [!NOTE]
> *Project Artifact Discrepancy:* The original `REPORT.md` provided in the repository states that LightGBM was the final model used. However, a rigorous inspection of the actual project code (`src/models/train_final_model.py` and `mlruns` database) confirms that **Random Forest** and **XGBoost** were implemented, and **RandomForestRegressor** was saved as the final production model. This report accurately reflects the *actual implemented code* rather than the textual claims in the original `REPORT.md`.

---

## SECTION 2 — BUSINESS PROBLEM

**Business Problem:** How can Rossmann ensure that stores have enough staff to handle customer footfall and enough inventory to prevent stockouts without overspending on excess labor or warehousing?
**Operational Problem:** Store managers manually predict daily sales up to six weeks in advance, leading to human bias, inconsistencies, and high error margins across the 3,000+ store network.
**Analytical Problem:** We need to identify historical sales patterns, the impact of holidays, the effect of promotions, and the influence of competitor proximity to estimate future demand mathematically.
**ML Problem:** Formulate a supervised regression task to predict the continuous variable `Sales` for a given `Store` and `Date` using historical multivariate time-series data.

---

## SECTION 3 — BUSINESS OBJECTIVES

* **Primary Objective:** Build a predictive model capable of forecasting daily store sales with high accuracy.
* **Secondary Objectives:** Identify key business drivers of sales (e.g., promotions, holidays, store types).
* **Business KPIs:** Decrease out-of-stock events, reduce surplus inventory costs, and optimize labor allocation efficiency.
* **ML KPIs:** Minimize Root Mean Square Percentage Error (RMSPE). RMSPE is crucial because it penalizes relative percentage errors rather than absolute errors, meaning a $100 error on a $500 sales day is penalized heavier than a $100 error on a $5000 sales day.
* **MLOps Objectives:** Ensure pipeline reproducibility through data versioning (DVC) and environment containerization (Docker).

---

## SECTION 4 — PROJECT SCOPE

**In Scope (Actually Implemented):**
* Data ingestion of raw CSV files.
* Data versioning using DVC.
* Data quality validation scripting.
* Extensive EDA in Jupyter Notebooks.
* Automated Feature Engineering (Lag, Rolling, Temporal).
* Time-based Train/Validation/Test splitting.
* Model Training (Random Forest, XGBoost, Baseline).
* Experiment tracking using MLflow.
* Containerized batch scoring using Docker.

**Out of Scope (Not Implemented):**
* Real-time API deployment (e.g., Flask/FastAPI).
* CI/CD pipelines (e.g., GitHub Actions).
* Automated Data/Model Drift monitoring.
* Hyperparameter optimization libraries like Optuna (only manual parameters used).
* Deep Learning approaches.

**Assumptions:**
* Future values for `Open`, `Promo`, and holidays are known 6 weeks in advance.
* Zero sales on days when a store is closed are valid structural zeros, not missing data.

---

## SECTION 5 — DATASET UNDERSTANDING

### Train Dataset (`train.csv`)
* **Rows:** 1,017,209 | **Columns:** 9
* **Date Range:** 2013-01-01 to 2015-07-31

| Column | Type | Meaning | Role |
| :--- | :--- | :--- | :--- |
| `Store` | int64 | Unique identifier for each store | Key |
| `DayOfWeek` | int64 | Day of the week (1=Monday, 7=Sunday) | Feature |
| `Date` | object | Date of the sales record | Temporal Key |
| `Sales` | int64 | Total turnover for the day | **Target** |
| `Customers` | int64 | Number of customers on the day | Feature / Leakage Risk |
| `Open` | int64 | Store operational status (0 = closed, 1 = open) | Feature |
| `Promo` | int64 | Indicates if store is running a promotion | Feature |
| `StateHoliday` | object | Indicates a state holiday (a, b, c, 0) | Feature |
| `SchoolHoliday` | int64 | Indicates if schools are closed | Feature |

### Store Dataset (`store.csv`)
* **Rows:** 1,115 | **Columns:** 10
* **Key Columns:** `StoreType` (a, b, c, d), `Assortment` (a, b, c), `CompetitionDistance` (meters to nearest competitor), `CompetitionOpenSince[Month/Year]`, `Promo2` (continuing and consecutive promotion).

---

## SECTION 6 — PROJECT ARCHITECTURE

```text
rossmann-sales/
├── data/
│   ├── predictions/       # Output from batch scoring container
│   ├── processed/         # Engineered feature datasets
│   ├── raw/               # Raw train.csv and store.csv (DVC tracked)
│   └── splits/            # Train, validation, test splits
├── mlruns/                # MLflow tracking backend
├── models/                # Serialized model artifacts (.pkl)
├── notebooks/             # Jupyter notebooks for EDA and prototyping
├── reports/               # Markdown reports (Data Quality, Business)
├── src/                   # Production Python source code
│   ├── data/              # Preprocessing, validation, and splitting scripts
│   ├── features/          # Feature engineering scripts
│   └── models/            # Training, evaluation, and prediction scripts
├── .dvc/                  # DVC configuration
├── Dockerfile             # Container definition for prediction
├── requirements.txt       # Python dependencies
└── REPORT.md              # Original (partially inaccurate) project summary
```

---

## SECTION 7 — DATA ENGINEERING

**Data Ingestion:** Data is read from local CSVs using Pandas.
**Raw Data Principle:** The raw `train.csv` and `store.csv` are NEVER modified directly. All processing creates new files in `data/processed/`.
**Data Cleaning Transformations (`src/data/preprocess_data.py`):**
* **Datatype Conversion:** `Date` converted from string to datetime objects.
* **Standardization:** `StateHoliday` contained mixed types (0, '0', '0.0', 'nan'). These were stripped and replaced with a uniform string "0".
* **Missing Values (Store):** Kept intact during basic processing, but filled with `0` during feature preparation prior to model training.
* **Merge Operation:** The pipeline performs a left join of `train` and `store` on the `Store` key, preserving the 1,017,209 row count.

---

## SECTION 8 — DATA QUALITY

The project includes an actual Data Quality Report (`reports/data_quality_report.md`):
* **Missing Values:** `train.csv` has 0 missing values. `store.csv` has missing values in `CompetitionDistance` (3), `CompetitionOpenSince...` (354), and `Promo2Since...` (544).
* **Duplicates:** 0 duplicate rows found.
* **Invalid Values:** No negative sales. 
* **Zero Sales:** 172,871 zero-sales records exist. Crucially, 172,817 of these occurred when `Open = 0`. Only 54 records had zero sales when the store was open. This confirms zero sales are mostly structural (store closed) and not data errors.

---

## SECTION 9 — DATA VALIDATION

The script `src/data/validate_data.py` enforces strict data contracts BEFORE any processing occurs:
* **Schema Validation:** Verifies exact column names match `EXPECTED_TRAIN_COLUMNS` and `EXPECTED_STORE_COLUMNS`.
* **Duplicate Validation:** Asserts `df.duplicated().sum() == 0`.
* **Entity Validation:** Asserts exactly 1,115 unique stores exist in both datasets.
* **Constraint Validation:** Asserts `Sales >= 0` and binary features (`Open`, `Promo`, `SchoolHoliday`) contain strictly 0 or 1.

---

## SECTION 10 — DATA VERSIONING WITH DVC

**Why DVC?** Git is designed for text and source code. Pushing large CSV datasets (like the 1M row train dataset) bloats the Git history and violates version control best practices.
**Implementation:** 
* `dvc add data/raw/train.csv` and `dvc add data/raw/store.csv` were executed.
* This generated `train.csv.dvc` and `store.csv.dvc` which contain MD5 hashes of the data. 
* The `.dvc` files are tracked by Git, while the actual CSVs are added to `.gitignore`.
* *Note:* If CSVs were previously tracked by Git, `git rm -r --cached data/raw/*.csv` is used to untrack them before adding to DVC.

---

## SECTION 11 — DATA PREPROCESSING PIPELINE

The `preprocess_data.py` script orchestrates the cleaning:
1. **Input:** `data/raw/train.csv` and `store.csv`
2. **Train Clean:** Fixes `Date` and standardizes `StateHoliday`.
3. **Store Clean:** Enforces numeric types for competition fields, strips strings in categorical fields.
4. **Merge:** Left join on `Store`.
5. **Validation:** Asserts no rows were lost (row count strictly 1,017,209).
6. **Output:** Saves to `data/processed/train_processed.csv`.

---

## SECTION 12 — EXPLORATORY DATA ANALYSIS

From `02_eda.ipynb` and `06_statistical_analysis.ipynb`:

**1. Promotion Impact**
* *Question:* Does promotion affect sales?
* *Output:* Non-promo mean: 4,406. Promo mean: 7,991.
* *Insight:* Promotions drive a massive **81.37% increase** in average sales.

**2. Day-of-Week Analysis**
* *Question:* Which days have highest sales?
* *Output:* Monday is highest (7,809). Sunday is lowest (204). 
* *Insight:* Sunday's average is 204 because the `Open` rate is only 2.48%. When open, Sunday sales actually average 8,224. `Open` status heavily confounds day-of-week averages.

**3. High Sales Outliers**
* *Question:* Should extreme values (up to 41,551) be capped?
* *Output:* High sales occur on open days, with large customer counts, across all store types. 
* *Decision:* Do not remove or cap extreme values; they represent valid peak demand.

**4. Sales vs Customers**
* *Question:* How correlated are they?
* *Output:* Correlation coefficient of 0.824.
* *Insight:* Strong linear relationship.

---

## SECTION 13 — STATISTICAL ANALYSIS

* **Mean vs Median:** Sales mean is 5773, median is 5744. The distribution is remarkably symmetrical for retail data, though long-tailed on the right.
* **Correlation vs Causation:** We observe that Sales are 81% higher on Promo days (Correlation). However, without controlled experiments, we cannot rule out that promotions are strategically scheduled on days with historically high demand (Causation).

---

## SECTION 14 — FEATURE ENGINEERING

Implemented in `src/features/build_features.py`:

| Feature | Original Source | Transformation | Why Created |
| :--- | :--- | :--- | :--- |
| `Year`, `Month`, `Day`, `WeekOfYear` | `Date` | Extracted datetime attributes | Captures annual, monthly, and weekly seasonality. |
| `HasCompetitionDistance` | `CompetitionDistance` | Binary flag (isnotna) | Models the absence of known competition. |
| `HasPromo2` | `Promo2` | Fillna & Binary flag | Indicates active recurring promotions. |
| `StoreAge` | `CompetitionOpen...` | `Year - CompetitionOpenSinceYear` (clipped > 0) | How established the competitor is. |
| `Sales_Lag_[1,7,14]` | `Sales` | `groupby('Store').shift(N)` | Autoregressive features capturing recent momentum. |
| `Sales_Rolling_Mean_[7,14,30]`| `Sales` | `shift(1).rolling(N).mean()` | Captures smoothed recent trend, avoiding leakage. |

---

## SECTION 15 — DATA LEAKAGE

**What is it?** Using information during training that will not be available at prediction time.
**Customers Variable Leakage:** While `Customers` correlates 0.82 with `Sales`, we cannot know exactly how many customers will visit a store 6 weeks from now. Therefore, `Customers` is safely EXCLUDED from the final feature set in `train_final_model.py`.
**Rolling Feature Leakage:** Notice the implementation: `shift(1).rolling(N).mean()`. The `shift(1)` is critical. It ensures today's sales are not included in today's rolling average calculation, which would be direct leakage.

---

## SECTION 16 — TRAIN/VALIDATION/TEST STRATEGY

Implemented in `src/data/split_data.py`.
**Strategy:** Strict Chronological Time-based Split. Random splitting would cause future data to leak into past predictions, destroying the time-series integrity.
* **Train:** 2013-01-01 to 2014-12-31
* **Validation:** 2015-01-01 to 2015-06-30 (used for model evaluation during development)
* **Test:** 2015-07-01 to 2015-07-31 (used as the final holdout to mimic a 1-month forecast)

---

## SECTION 17 — BASELINE MODEL

Implemented in `src/models/baseline.py`.
* **Model:** Naive Lag-7 model (Predicts that today's sales will equal sales from exactly 7 days ago).
* **Metric:** RMSPE.
* **Test Performance:** 0.3842.
* **Why?** A machine learning model is only valuable if it beats a simple, free heuristic. Our advanced models must achieve an RMSPE significantly lower than 0.3842.

---

## SECTION 18 & 19 — MODEL DEVELOPMENT & MATHEMATICS

**1. Random Forest (`train_random_forest.py` & `train_final_model.py`)**
* *Algorithm:* An ensemble of decision trees trained on random subsets of data and features (bagging). 
* *Mathematics:* Minimizes variance by averaging multiple high-variance trees. Splits are determined by maximizing information gain (reducing variance of the target variable).
* *Parameters used:* `n_estimators=100, max_depth=20, min_samples_leaf=2`.

**2. XGBoost (`train_xgboost.py`)**
* *Algorithm:* eXtreme Gradient Boosting.
* *Mathematics:* A sequential ensemble (boosting). Each tree attempts to predict the residual errors of the previous trees. It uses a regularized objective function to prevent overfitting.
* *Parameters used:* `n_estimators=500, learning_rate=0.05, max_depth=8`.

---

## SECTION 20 & 21 — MODEL EVALUATION & COMPARISON

**Metric Used: RMSPE (Root Mean Square Percentage Error)**
* *Formula:* `sqrt(mean(((Actual - Predicted) / Actual)^2))`
* *Why:* Standard metrics like RMSE are heavily skewed by stores with massive sales volumes. RMSPE treats a 10% error on a $1,000 day identically to a 10% error on a $10,000 day.

**Actual Results (from MLflow DB):**
| Model | Validation RMSPE | Test RMSPE | Improvement over Baseline |
| :--- | :--- | :--- | :--- |
| Baseline (Lag-7) | - | 0.3842 | - |
| Random Forest | 0.1578 | **0.1199** | **68.78%** |

*Final Selection:* The Random Forest model (`train_final_model.py`) was chosen as the final production model.

---

## SECTION 22 — HYPERPARAMETER TUNING

*Status:* **Partially Implemented (Manual tuning).**
The project relies on hardcoded, mathematically sound hyperparameters in the training scripts (e.g., setting `max_depth=20` to prevent infinite tree growth, and `min_samples_leaf=2` to prevent overfitting on single observations). Automated search frameworks like Optuna or GridSearchCV are not present in the codebase.

---

## SECTION 23 & 24 — EXPERIMENT TRACKING & VERSIONING

**MLflow Implementation (`mlflow_tracking.py`):**
* **Experiment Name:** `rossmann-sales`
* **Run Name:** `random-forest-final-metrics`
* **Logged Params:** `n_estimators`, `max_depth`, `min_samples_leaf`, `training_rows`, `feature_count`.
* **Logged Metrics:** `validation_rmspe`, `test_rmspe`, `baseline_test_rmspe`, `improvement_percent`.
* **Model Registry:** The trained model object is logged using `mlflow.sklearn.log_model()`.

**Model Versioning:** The final model artifact is serialized locally via Joblib to `models/random_forest_final.pkl`.

---

## SECTION 25 — MLOPS ARCHITECTURE

| Stage | Status | Implementation |
| :--- | :--- | :--- |
| Data Versioning | ✅ Implemented | DVC tracking `raw/*.csv` |
| Data Validation | ✅ Implemented | `validate_data.py` schema checks |
| Model Tracking | ✅ Implemented | MLflow SQLite DB |
| Model Training | ✅ Implemented | Python scripts (`src/models/`) |
| Containerization | ✅ Implemented | Dockerfile for prediction |
| CI/CD Pipeline | ❌ Not Implemented | No `.github/workflows` found |
| API Deployment | ❌ Not Implemented | Batch scoring only |
| Drift Monitoring | ❌ Not Implemented | No EvidentlyAI/Prometheus |

---

## SECTION 26 — DOCKER / CONTAINERIZATION

The project packages the batch scoring system into a Docker container.
* **Base Image:** `python:3.11-slim` (Lightweight Debian base).
* **Dependencies:** Installed via `pip install -r requirements.txt`.
* **Volume/Copy:** Copies `src/`, `data/processed/`, and `data/splits/`.
* **Execution:** Sets `CMD ["python", "src/models/predict_production.py"]` to automatically run batch inference upon container startup and output to `data/predictions/submission.csv`.

---

## SECTION 27 TO 30 — MISSING COMPONENTS

* **CI/CD:** Not implemented. A production setup would require Jenkins or GitHub Actions to automate running `validate_data.py` on PRs.
* **Testing:** No `pytest` unit tests for individual functions exist. Only data validation scripts exist.
* **Deployment:** No real-time API. Predictions are strictly batch-based via `predict_production.py`.
* **Monitoring:** Data/Concept drift monitoring is absent.

---

## SECTION 31 — FINAL SOLUTION 

**Pipeline Flow:**
Raw Data (DVC) → `validate_data.py` → `preprocess_data.py` → `build_features.py` → `split_data.py` → `train_final_model.py` (Random Forest) → Serialized `.pkl` → Dockerized `predict_production.py` → Batch Output CSV.

**Final Result:** A robust, automated pipeline capable of predicting 6 weeks of sales with a ~12% relative error margin (RMSPE = 0.1199), representing a ~69% improvement over the baseline.

---

## SECTION 32 — BUSINESS INTERPRETATION

1. **Promo Strategies:** Promotions generate an immediate 81% spike. Rossmann should avoid overlapping promos on historically busy days (like Mondays) and instead use them to lift slow mid-week days.
2. **Staffing Optimization:** Since Sundays are dead (mostly closed) and Mondays are peak (7809 avg sales), labor should be heavily rotated to staff Monday morning shifts.
3. **School Holidays:** Result in a ~15% sales bump. Inventory logic for snacks/family items should be adjusted locally based on school calendars.

---

## SECTION 33 & 34 — LIMITATIONS & FUTURE IMPROVEMENTS

**Limitations:**
* We assumed `Open` status is perfectly known in advance. Unexpected closures (e.g., strikes, weather) will ruin the forecast.
* Missing macroeconomic data (inflation, local GDP) limits the model's understanding of long-term economic shifts.
* `RandomForest` is heavy. A 20-depth forest with 100 trees is large in memory compared to boosting algorithms.

**Future Improvements:**
* Implement automated hyperparameter tuning (Optuna).
* Deploy the model as a REST API using FastAPI.
* Implement CI/CD using GitHub Actions.

---

## SECTION 35 & 48 — END-TO-END ARCHITECTURE DIAGRAM

```text
       ┌──────────────────┐
       │  Business Need   │ (Predict 6-week sales)
       └────────┬─────────┘
                ↓
       ┌──────────────────┐
       │    Raw Data      │ (train.csv, store.csv tracked by DVC)
       └────────┬─────────┘
                ↓
       ┌──────────────────┐
       │ Data Validation  │ (validate_data.py schema/dupe checks)
       └────────┬─────────┘
                ↓
       ┌──────────────────┐
       │ Data Processing  │ (preprocess_data.py -> train_processed.csv)
       └────────┬─────────┘
                ↓
       ┌──────────────────┐
       │ Feature Engineer │ (build_features.py -> lag/rolling/dates)
       └────────┬─────────┘
                ↓
       ┌──────────────────┐
       │ Data Splitting   │ (split_data.py -> Time-based train/val/test)
       └────────┬─────────┘
                ↓
       ┌──────────────────┐
       │ Model Training   │ (train_final_model.py -> Random Forest)
       └────────┬─────────┘
                ↓
       ┌──────────────────┐
       │ Expermt Tracking │ (mlflow_tracking.py -> SQLite DB)
       └────────┬─────────┘
                ↓
       ┌──────────────────┐
       │     Docker       │ (Dockerfile -> predict_production.py)
       └──────────────────┘
```

---

## SECTION 36 — FILE-BY-FILE TECHNICAL WALKTHROUGH

| File | Purpose | Key Logic |
| :--- | :--- | :--- |
| `validate_data.py` | Data Contracts | `assert` statements checking schema, duplicates, and constraints (`Sales >= 0`). |
| `preprocess_data.py` | Cleaning | Date parsing, StateHoliday standardizing, Left Join on Store. |
| `build_features.py` | Feature Eng | Extracts Date parts. Uses `groupby.shift(1).rolling(N)` to build safe lag features. |
| `split_data.py` | Train/Test Split | Chronological split. Train < 2015, Val H1 2015, Test Jul 2015. |
| `baseline.py` | Benchmark | Creates a naive `Sales_Lag_7` model to calculate baseline RMSPE (0.3842). |
| `train_final_model.py`| Model Training | Fits `RandomForestRegressor`, handles categorical dummies, outputs `.pkl`. |
| `mlflow_tracking.py` | Logging | Logs params and metrics to MLflow, showing 68% improvement. |

---

## SECTION 39 — VIVA QUESTIONS & ANSWERS (CHEAT SHEET)

**Q: Why did you use RMSPE instead of RMSE?**
A: Retail stores vary wildly in volume. A Type B store averages 10k sales, a Type D averages 5.6k. RMSE penalizes absolute errors, so it would overly focus the model on high-volume stores. RMSPE penalizes relative percentage errors, treating all stores fairly.

**Q: Why didn't you use `train_test_split` from sklearn?**
A: `train_test_split` shuffles data randomly. This is a time-series problem. A random split would put future data in the training set to predict past data in the test set, causing massive data leakage. We used a strict chronological cutoff in `split_data.py`.

**Q: How did you calculate rolling averages without causing data leakage?**
A: In `build_features.py`, we used `groupby('Store')['Sales'].transform(lambda x: x.shift(1).rolling(N).mean())`. The `shift(1)` is crucial—it pushes the data down by one day so that today's sales are NOT included in today's rolling average.

**Q: Why did you include `Open` status? Isn't it obvious zero sales happen when closed?**
A: Yes, but the model needs to learn *why* Sunday sales are low. Without `Open`, the model just learns "Sunday = bad". With `Open`, it learns "Sunday = bad *because* Open = 0". When a store actually opens on a Sunday, it averages 8.2k sales!

**Q: What is DVC and why not just use Git?**
A: Git tracks text diffs. Our `train.csv` is over 1M rows. Pushing this to Git bloats the repository and makes cloning impossibly slow. DVC creates an MD5 hash pointer (`train.csv.dvc`) that Git tracks, while DVC handles the actual large file storage.

**Q: Your `REPORT.md` mentioned LightGBM, but what did you actually deploy?**
A: While LightGBM may have been evaluated historically, the actual final code (`train_final_model.py`) trains a Random Forest Regressor, and the MLflow database verifies this model achieved the 0.1199 test RMSPE.

**Q: Why did you drop the `Customers` column before training?**
A: Because at the moment of prediction (6 weeks in advance), we do not know how many customers will walk in. Using it as a feature would be a classic example of data leakage.

---

## SECTION 47 — REPRODUCIBILITY GUIDE

To reproduce this exact project on a local machine:

1. `git clone <repository_url>`
2. `cd rossmann-sales`
3. `python -m venv .venv`
4. `source .venv/bin/activate` (or `.\.venv\Scripts\activate` on Windows)
5. `pip install -r requirements.txt`
6. `dvc pull` (assuming remote is configured, otherwise ensure raw CSVs are in `data/raw/`)
7. `python src/data/validate_data.py`
8. `python src/data/preprocess_data.py`
9. `python src/features/build_features.py`
10. `python src/data/split_data.py`
11. `python src/models/train_final_model.py`
12. `python src/models/mlflow_tracking.py`
13. `mlflow ui` (to view metrics locally)
14. `docker build -t rossmann-sales .`
15. `docker run -v $(pwd)/data:/app/data rossmann-sales` (to execute batch predictions)
