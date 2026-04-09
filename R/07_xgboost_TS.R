# =============================================================================
# 07_xgboost_TS — XGBOOST: TEMPORAL SPLIT (TS) SCHEME
# Project: RPE Prediction in Competitive Runners
# =============================================================================

# --- Shared XGBoost hyperparameters -----------------------------------------
# Identical in both TS and FW scripts so any difference in final test metrics
# is attributable only to the validation scheme, not to model configuration.

xgb_params_TS <- list(
  objective        = "reg:squarederror",
  eta              = 0.05,      # learning rate — most important hyperparameter
  max_depth        = 4,
  subsample        = 0.8,
  colsample_bytree = 0.8,
  lambda           = 1,         # L2 regularisation (default)
  alpha            = 0          # L1 regularisation (none)
)

# --- DMatrix objects for the TS split ----------------------------------------
dtrain_TS <- xgb.DMatrix(data = as.matrix(X_train_TS), label = y_train_TS)
dtest_TS  <- xgb.DMatrix(data = as.matrix(X_test_TS),  label = y_test_TS)

# --- nrounds selection via random k-fold CV on training data -----------------
# xgb.cv uses random 5-fold here solely to determine the early-stopping point
# for nrounds. Random k-fold (not timeslice) is standard practice for this
# step; the definitive temporal evaluation is on the held-out test set.

set.seed(42)
xgb_cv_TS <- xgb.cv(
  params                = xgb_params_TS,
  data                  = dtrain_TS,
  nrounds               = 1000,
  nfold                 = 5,
  metrics               = "rmse",
  early_stopping_rounds = 50,
  verbose               = 0
)

best_nrounds_TS <- xgb_cv_TS$early_stop$best_iteration   # xgboost >= 2.0 API
message("[TS] XGBoost best nrounds: ", best_nrounds_TS)

# --- Fit final XGBoost model on the TS training set -------------------------
set.seed(42)
xgb_model_TS <- xgb.train(
  params  = xgb_params_TS,
  data    = dtrain_TS,
  nrounds = best_nrounds_TS,
  verbose = 0
)

# --- Predict on held-out test set --------------------------------------------
pred_xgb_TS <- predict(xgb_model_TS, dtest_TS)

# --- Variable importance (Gain) ----------------------------------------------
xgb_imp_df_TS <- xgb.importance(model = xgb_model_TS) %>%
  as.data.frame()

message("[TS] XGBoost fitted. Top feature: ", xgb_imp_df_TS$Feature[1])
