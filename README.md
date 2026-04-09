# Tree-Based RPE Prediction in Competitive Runners

Predicting the Rating of Perceived Exertion (RPE) from the preceding six days of training history using Decision Tree, Random Forest, and XGBoost models.

## Dataset

The dataset was collected by Lovdal et al. (2021) and is publicly available on Kaggle:

**Download:** https://www.kaggle.com/code/mohamedbakrey/anticipate-injured-numbers-in-competitive-runners

It contains seven years of daily training logs from 74 competitive middle- and long-distance runners. Place the file `day_approach_maskedID_timeseries.csv` in the project root before running the code.

## Requirements

The following R packages are required:

- `tidyverse`
- `readr`
- `rpart` and `rpart.plot`
- `randomForest`
- `xgboost`
- `caret`
- `Metrics`

Install them with:

```r
install.packages(c("tidyverse", "readr", "rpart", "rpart.plot",
                    "randomForest", "xgboost", "caret", "Metrics"))
```

## How to run

1. Open R or RStudio with the working directory set to the project root.
2. Run the full pipeline:

```r
source("main.R")
```

This executes, in order:
- Data loading and cleaning
- Feature engineering (lag variables from t-1 to t-6)
- Train/test split under two validation schemes (Temporal Split and Walk-Forward CV)
- Model training and hyperparameter tuning (Decision Tree, Random Forest, XGBoost)
- Results aggregation and metric computation
- Figure generation (saved as PDF and PNG in `figures/`)

All outputs are reproducible (`set.seed(42)` is set at the top of `main.R`).

## Project structure

```
main.R                 Entry point
R/
  00_packages.R        Library loading
  01_load_clean.R      Data import and filtering
  02_features.R        Feature selection (lag columns)
  02b_eda_figures.R    Exploratory data analysis figures
  03_split_TS.R        Temporal Split scheme
  03_split_FW.R        Walk-Forward CV scheme
  05_decision_tree_TS.R / _FW.R
  06_random_forest_TS.R / _FW.R
  07_xgboost_TS.R / _FW.R
  08_results.R         Metrics aggregation
  09_figures_TS.R      TS scheme figures
  09_figures_FW.R      FW scheme figures
figures/               Generated plots (PDF + PNG)
report/                LaTeX source for the article
```

## Reference

Lovdal, S., Den Hartigh, R., & Azzopardi, G. (2021). Injury prediction in competitive runners with machine learning. *International Journal of Sports Physiology and Performance*, 16(10), 1507-1514.
