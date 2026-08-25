# Rossmann Sales Forecasting: Project Report

## 1. Project Overview & Executive Summary

### 1.1 Business Objective
Retail chain performance heavily relies on accurate demand planning. Rossmann operates over 3,000 drug stores across 7 European countries. Store managers are frequently tasked with predicting daily sales up to six weeks in advance. Accurate sales forecasts enable:
* Optimized staff scheduling to match customer footfall.
* Efficient inventory management, minimizing stockouts and holding costs.
* Data-driven promotional planning across store networks.

### 1.2 Key Deliverables & Results
* **Predictive Pipeline:** Developed a production-grade machine learning model using **LightGBM** optimized for tabular retail data.
* **Feature Engineering:** Built a robust feature extraction pipeline handling temporal elements, competition metrics, and promotional intervals.
* **Containerized Inference:** Packaged the prediction pipeline inside a lightweight **Docker** container ensuring portable, reproducible batch scoring.
* **Version Control:** Managed code lifecycle and versioning using **Git**.

---

## 2. Exploratory Data Analysis & Feature Engineering

### 2.1 Data Architecture
The dataset comprises historical sales, promotional information, and store metadata across two primary data sources:
1. **Sales / Train / Test Data (`train.csv`, `test.csv`):** Contains daily records of store sales, customer counts, open status, holiday indicators, and promotion types.
2. **Store Metadata (`store.csv`):** Contains static and dynamic store-level features including store type, assortment level, competition distance, competition opening timelines, and `Promo2` recurring promotion details.

### 2.2 Data Hygiene & Preprocessing
To prepare the raw datasets for modeling, several preprocessing steps were executed:
* **Missing Value Treatment:**
  * `CompetitionDistance`: Imputed missing values with twice the maximum observed distance to signify the absence of immediate local competition.
  * `CompetitionOpenSinceMonth` / `Year` & `Promo2SinceWeek` / `Year`: Filled missing structural timestamps with `0` where promotions or competitions were non-existent.
  * `PromoInterval`: Replaced missing string categories with empty strings to prevent string-handling exceptions during encoding.
* **Temporal Sorting & Filtering:** Filtered out records where stores were closed (`Open == 0`) or recorded zero sales to prevent model bias during training.

### 2.3 Feature Engineering Highlights
To maximize predictive power, the pipeline extracts several high-value predictive signals:
* **Date-Part Decomposition:** Extracted granular temporal attributes from date fields including `Year`, `Month`, `Day`, `DayOfWeek`, and `WeekOfYear` to capture seasonality and weekly consumer purchasing trends.
* **Competition Tenure Metrics:** Calculated active competition duration variables representing how long a competing store has been open relative to the sales date.
* **Promotion Indicators:** Encoded recurring promotional campaigns (`Promo2`) by cross-referencing active promotion calendar months against store timestamps.

---

## 3. Modeling & Experimentation Strategy

### 3.1 Model Selection & Justification
For the Rossmann sales forecasting task, **LightGBM** (Light Gradient Boosting Machine) was selected as the primary modeling engine. Key reasons include:
* **Handling Tabular Data:** Gradient boosted decision trees consistently outperform deep learning and linear models on structured, heterogeneous retail tabular data.
* **Speed & Scalability:** LightGBM trains significantly faster and consumes less memory than traditional algorithms like XGBoost while maintaining high predictive accuracy.
* **Missing Value Handling:** Native support for handling sparse and missing values effectively without complex imputation models.

### 3.2 Validation & Evaluation Metrics
* **Evaluation Metric:** **RMSPE** (Root Mean Square Percentage Error) was chosen as the core metric, directly penalizing percentage deviations in sales predictions and mirroring the business competition metric.
* **Temporal Cross-Validation:** Ensured strict chronological splitting to prevent data leakage, mimicking real-world future prediction constraints.

---

## 4. MLOps, Containerization & Deployment

### 4.1 Reproducibility & Serialization
* **Model Artifacts:** Trained models were serialized using Python's `joblib` library (`best_model.pkl`) to retain internal hyperparameters, tree structures, and feature mappings reliably.
* **Experiment Tracking:** Logged experimental parameters and configurations to maintain clean lineage from training runs to production artifacts.

### 4.2 Containerization with Docker
To guarantee identical execution environments across local development and production servers:
* **Base Image:** Built on top of a lightweight Debian-based `python:3.10-slim` image.
* **System Dependencies:** Explicitly installed `libgomp1` (OpenMP runtime library) to support C-level multithreading required by LightGBM on Linux containers.
* **Execution Workflow:** Automated batch scoring (`predict.py`) triggered automatically upon container startup, depositing final predictions at `data/predictions/submission.csv`.

### 4.3 Version Control
* **Git Repository Initialization:** Structured project tracking with Git (`main` branch) to version control source code, configuration scripts, and documentation assets.