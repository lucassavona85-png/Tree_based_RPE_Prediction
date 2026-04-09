# =============================================================================
# MAIN — RPE Prediction Pipeline Entry Point (Dual Validation Scheme)
# Project: RPE Prediction in Competitive Runners
# =============================================================================
# Two validation schemes are run side by side:
#   TS — Temporal Split: fit once on 80 %, evaluate once on test 20 %
#   FW — Walk-Forward CV: 5 expanding folds on training set, same held-out test
# Both use identical features, hyperparameters, and test set.

# Global seed for full reproducibility across all stochastic model steps
set.seed(42)

source("R/00_packages.R")
source("R/01_load_clean.R")
source("R/02_features.R")

# --- Scheme 1: Temporal Split ------------------------------------------------
source("R/03_split_TS.R")
source("R/02b_eda_figures.R")    # EDA figures (RPE histogram + correlation heatmap)
source("R/05_decision_tree_TS.R")
source("R/06_random_forest_TS.R")
source("R/07_xgboost_TS.R")

# --- Scheme 2: Walk-Forward CV -----------------------------------------------
source("R/03_split_FW.R")
source("R/05_decision_tree_FW.R")
source("R/06_random_forest_FW.R")
source("R/07_xgboost_FW.R")

# --- Combined results and figures -------------------------------------------
source("R/08_results.R")
source("R/09_figures_TS.R")
source("R/09_figures_FW.R")
