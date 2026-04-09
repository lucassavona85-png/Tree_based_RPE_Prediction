# =============================================================================
# 07_xgboost_FW — XGBOOST: WALK-FORWARD CV (FW) SCHEME
# Project: RPE Prediction in Competitive Runners
# =============================================================================

# --- Shared XGBoost hyperparameters -----------------------------------------
# Identical to xgb_params_TS to ensure any difference in test metrics reflects
# the validation scheme, not the model configuration.

xgb_params_FW <- list(
  objective        = "reg:squarederror",
  eta              = 0.05,
  max_depth        = 4,
  subsample        = 0.8,
  colsample_bytree = 0.8,
  lambda           = 1,
  alpha            = 0
)

# --- nrounds (aliased from TS) -----------------------------------------------
# TS and FW train on identical data; xgb.cv would return the same stopping
# point. Reuse best_nrounds_TS to avoid a redundant 1000-round CV pass.
best_nrounds_FW <- best_nrounds_TS
message("[FW] XGBoost best nrounds (reuses TS result): ", best_nrounds_FW)

# --- Walk-forward cross-validation loop for model selection reporting --------
# Each fold trains on the expanding chronological window and predicts on the
# following wf_horizon_FW rows. nrounds is fixed at best_nrounds_FW to avoid
# double-nested CV while keeping model complexity constant across folds.

xgb_fold_rmse_FW <- numeric(n_folds_FW)
for (i in seq_len(n_folds_FW)) {
  idx_tr <- wf_slices_FW$train[[i]]
  idx_te <- wf_slices_FW$test[[i]]

  d_fold_tr <- xgb.DMatrix(
    as.matrix(X_train_s_FW[idx_tr, ]), label = y_train_s_FW[idx_tr]
  )
  d_fold_te <- xgb.DMatrix(
    as.matrix(X_train_s_FW[idx_te, ]), label = y_train_s_FW[idx_te]
  )

  set.seed(42)
  fold_xgb_FW <- xgb.train(
    params  = xgb_params_FW,
    data    = d_fold_tr,
    nrounds = best_nrounds_FW,
    verbose = 0
  )
  fold_pred_xgb_FW    <- predict(fold_xgb_FW, d_fold_te)
  xgb_fold_rmse_FW[i] <- Metrics::rmse(y_train_s_FW[idx_te], fold_pred_xgb_FW)
  message("[FW] XGB fold ", i, "/", n_folds_FW,
          " RMSE: ", round(xgb_fold_rmse_FW[i], 4))
}

xgb_cv_mean_FW <- round(mean(xgb_fold_rmse_FW), 4)
xgb_cv_sd_FW   <- round(sd(xgb_fold_rmse_FW),   4)
message("[FW] XGBoost walk-forward CV RMSE: ", xgb_cv_mean_FW, " \u00b1 ",
        xgb_cv_sd_FW, " (", n_folds_FW, " folds)")

# --- Final model (aliased from TS) ------------------------------------------
# TS and FW use the same training data and hyperparameters; the final fitted
# model and predictions are therefore identical. Aliasing avoids training a
# second full model while xgb_model_TS is already resident in memory.
xgb_model_FW  <- xgb_model_TS
pred_xgb_FW   <- pred_xgb_TS
xgb_imp_df_FW <- xgb_imp_df_TS

message("[FW] XGBoost: reuses TS model (same training data and params).")
message("[FW] XGBoost fitted. Top feature: ", xgb_imp_df_FW$Feature[1])
