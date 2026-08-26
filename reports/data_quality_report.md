# Data Quality Report

## 1. Dataset Overview

### Train Dataset
- Number of rows: 1017209
- Number of columns: 9
- Date range: 2013-01-01   to    2015-07-31
- Number of unique stores: 1115

### Store Dataset
- Number of rows: 1115
- Number of columns: 10
- Number of unique stores: 1115

## 2. Missing Values

### Train
- Columns with missing values: 0
- Missing-value counts: 0

### Store
- Columns with missing values: CompetitionDistance , CompetitionOpenSinceMonth, CompetitionOpenSinceYear,Promo2SinceWeek,Promo2SinceYear,PromoInterval
- Missing-value counts: 3,354,354,544,544,544

## 3. Duplicates

### Train
- Duplicate rows: 0

### Store
- Duplicate rows: 0

## 4. Data Type Issues

- Columns requiring conversion:
  - Train `Date` should be converted from object/string to datetime.
  - Store `CompetitionOpenSinceMonth` and `CompetitionOpenSinceYear` should be treated as numeric/date-related fields.
  - Store `Promo2SinceWeek` and `Promo2SinceYear` should be treated as numeric/date-related fields.

- Columns requiring further investigation:
  - `StateHoliday` contains mixed representations and requires standardization.
  - Missing values in the Store dataset require business-based treatment rather than automatic deletion.
  - `PromoInterval` requires conversion into meaningful promotional features.

## 5. Initial Data Quality Concerns

- Missing values:
    - The Store dataset contains missing values in competition and Promo2-related fields.
    - These missing values may represent "not applicable" rather than bad data and require business-based treatment.

- Duplicates:
    - No duplicate rows were identified in either dataset.

- Inconsistent data types:
    - `Train.Date` requires conversion to datetime.
    - `StateHoliday` requires standardization.

- Potential outliers:
    - Sales and Customers may contain extreme values.
    - Outlier investigation will be performed during EDA rather than removing observations at this stage.


- Other observations:
    - The Train dataset contains records for 1,115 stores.
    - The Store dataset also contains 1,115 stores.
    - Store-level data can therefore be joined using the `Store` key.
    - The raw datasets will be preserved unchanged.

## 6. Decision

The raw datasets will NOT be modified.

Cleaning and transformation will be performed in a separate
data-processing pipeline, preserving the original raw data.