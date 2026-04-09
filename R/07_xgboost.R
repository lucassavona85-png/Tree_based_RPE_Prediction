# =============================================================================
# 07 — XGBOOST
# Project: RPE Prediction in Competitive Runners
# =============================================================================

# --- Shared model hyperparameters --------------------------------------------
# Defined once and reused by xgb.cv, the walk-forward fold loop, and the
# final model to guarantee identical configuration across all three stages.

xgb_params <- list(
  objective        = "reg:squarederror",
  eta              = 0.05,      # learning rate — most important hyperparameter
  max_depth        = 4,
  subsample        = 0.8,
  colsample_bytree = 0.8,
  lambda           = 1,         # L2 regularisation (default)
  alpha            = 0          # L1 regularisation (none)
)

# --- Build XGBoost DMatrix objects -------------------------------------------
# DMatrix is XGBoost's internal data structure; column names from X_train
# are preserved and will appear in the importance table.

dtrain <- xgb.DMatrix(data = as.matrix(X_train), label = y_train)
dtest  <- xgb.DMatrix(data = as.matrix(X_test),  label = y_test)

# --- Cross-validation to select optimal number of boosting rounds -----------
# eta (learning rate) is the most important hyperparameter for gradient boosting:
# a lower value yields more trees but better generalisation.
# nrounds is determined automatically via early stopping on the CV error:
# training stops when the test RMSE has not improved for 50 consecutive rounds.
# Random k-fold is used here solely to find the stopping point; the definitive
# temporal evaluation follows in the walk-forward fold loop below.

set.seed(42)
xgb_cv <- xgb.cv(
  params                = xgb_params,
  data                  = dtrain,
  nrounds               = 1000,
  nfold                 = 5,
  metrics               = "rmse",
  early_stopping_rounds = 50,
  verbose               = 0
)

# In xgboost >= 2.0 the best iteration is nested under $early_stop
best_nrounds <- xgb_cv$early_stop$best_iteration
message("XGBoost best nrounds (early stopping): ", best_nrounds)

# --- Walk-forward cross-validation for model selection reporting -------------
# Each fold trains on the expanding chronological window from wf_slices
# (defined in 03_split.R, applied to X_train_s sorted by Date) and predicts
# on the next wf_horizon rows. nrounds is fixed at best_nrounds throughout to
# avoid double-nested CV while keeping model complexity constant across folds.

xgb_fold_rmse <- numeric(n_folds)
for (i in seq_len(n_folds)) {
  idx_tr <- wf_slices$train[[i]]
  idx_te <- wf_slices$test[[i]]

  d_fold_tr <- xgb.DMatrix(as.matrix(X_train_s[idx_tr, ]), label = y_train_s[idx_tr])
  d_fold_te <- xgb.DMatrix(as.matrix(X_train_s[idx_te, ]), label = y_train_s[idx_te])

  set.seed(42)
  fold_model        <- xgb.train(params  = xgb_params,
                                  data    = d_fold_tr,
                                  nrounds = best_nrounds,
                                  verbose = 0)
  fold_pred         <- predict(fold_model, d_fold_te)
  xgb_fold_rmse[i]  <- Metrics::rmse(y_train_s[idx_te], fold_pred)
  message("XGB fold ", i, "/", n_folds, " RMSE: ", round(xgb_fold_rmse[i], 4))
}

xgb_cv_mean <- round(mean(xgb_fold_rmse), 4)
xgb_cv_sd   <- round(sd(xgb_fold_rmse),   4)
message("XGBoost walk-forward CV RMSE: ", xgb_cv_mean, " \u00b1 ", xgb_cv_sd,
        " (", n_folds, " folds)")

# --- Fit final XGBoost model -------------------------------------------------
# Trained on the full training DMatrix with the hyperparameters and nrounds
# found by cross-validation above.

set.seed(42)
xgb_model <- xgb.train(
  params  = xgb_params,
  data    = dtrain,
  nrounds = best_nrounds,
  verbose = 0
)

# --- Predict on test set -----------------------------------------------------
xgb_pred <- predict(xgb_model, dtest)

# --- Variable importance (Gain) ----------------------------------------------
# Gain: average improvement in the loss function brought by a feature across
# all splits where it is used. Preferred over Cover or Frequency for
# interpretability, and is the native importance measure for gradient boosting.

xgb_imp_df <- xgb.importance(model = xgb_model) %>%
  as.data.frame()

message("XGBoost fitted. Top feature: ", xgb_imp_df$Feature[1])
