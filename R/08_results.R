# =============================================================================
# 08 — RESULTS TABLE: BOTH VALIDATION SCHEMES COMBINED
# Project: RPE Prediction in Competitive Runners
# =============================================================================

# --- Evaluation helpers ------------------------------------------------------
# R² is computed as the coefficient of determination: 1 - SS_res / SS_tot.
# This formula is valid for any predictor (including constants like the dummy)
# and can be negative when the model performs worse than predicting the mean.

r_squared <- function(y_true, y_pred) {
  ss_res <- sum((y_true - y_pred)^2)
  ss_tot <- sum((y_true - mean(y_true))^2)
  round(1 - ss_res / ss_tot, 3)
}

eval_model <- function(y_true, y_pred, name) {
  tibble(
    Model = name,
    RMSE  = round(Metrics::rmse(y_true, y_pred), 3),
    MAE   = round(Metrics::mae(y_true,  y_pred), 3),
    R2    = r_squared(y_true, y_pred)
  )
}

# --- Dummy baseline ----------------------------------------------------------
# Predicts the training-set mean for every test observation.
# Both schemes share the same test set; training means are identical since
# train_TS and train_FW cover the same Date range.

dummy_pred_TS <- rep(mean(y_train_TS), length(y_test_TS))
dummy_pred_FW <- rep(mean(y_train_FW), length(y_test_FW))

# --- Scheme 1: Temporal Split (TS) results -----------------------------------
# Models are fitted once on the first 80 % of the data (by Date) and evaluated
# once on the last 20 %. No cross-validation is used during training.

results_TS <- bind_rows(
  eval_model(y_test_TS, dummy_pred_TS, "Dummy Baseline"),
  eval_model(y_test_TS, pred_dt_TS,    "Decision Tree"),
  eval_model(y_test_TS, pred_rf_TS,    "Random Forest"),
  eval_model(y_test_TS, pred_xgb_TS,   "XGBoost")
)

message("\n=== TS \u2014 Final Performance on Held-Out Test Set ===")
message("Note: RPE is on a 0\u20131 normalised scale.")
print(results_TS)

# --- Scheme 2: Walk-Forward CV (FW) — fold-level RMSE -----------------------
# One row per fold per model; used for the fig_cv_folds_FW line plot.

cv_results_FW <- bind_rows(
  tibble(Model = "Decision Tree",
         Fold  = seq_len(n_folds_FW),
         RMSE  = dt_fold_rmse_FW),
  tibble(Model = "Random Forest",
         Fold  = seq_len(n_folds_FW),
         RMSE  = rf_fold_rmse_FW),
  tibble(Model = "XGBoost",
         Fold  = seq_len(n_folds_FW),
         RMSE  = xgb_fold_rmse_FW)
)

# Summary: mean \u00b1 sd per model across folds
cv_summary_FW <- cv_results_FW %>%
  group_by(Model) %>%
  summarise(
    CV_RMSE    = round(mean(RMSE), 4),
    CV_RMSE_SD = round(sd(RMSE),   4),
    .groups = "drop"
  ) %>%
  mutate(CV_RMSE_fmt = paste0(CV_RMSE, " \u00b1 ", CV_RMSE_SD))

message("\n=== FW \u2014 Walk-Forward CV RMSE (", n_folds_FW, " folds, mean \u00b1 sd) ===")
print(cv_summary_FW %>% select(Model, CV_RMSE_fmt))

# --- Scheme 2: Walk-Forward CV (FW) — final test results --------------------
# Models fitted on the full FW training set are evaluated on the same held-out
# test set as TS, enabling a direct comparison of both schemes.

results_FW <- bind_rows(
  eval_model(y_test_FW, dummy_pred_FW, "Dummy Baseline"),
  eval_model(y_test_FW, pred_dt_FW,    "Decision Tree"),
  eval_model(y_test_FW, pred_rf_FW,    "Random Forest"),
  eval_model(y_test_FW, pred_xgb_FW,   "XGBoost")
)

message("\n=== FW \u2014 Final Performance on Held-Out Test Set ===")
message("Note: RPE is on a 0\u20131 normalised scale.")
print(results_FW)
