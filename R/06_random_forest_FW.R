# =============================================================================
# 06_random_forest_FW — RANDOM FOREST: WALK-FORWARD CV (FW) SCHEME
# Project: RPE Prediction in Competitive Runners
# =============================================================================

# --- Walk-forward cross-validation loop for model selection reporting --------
# Each fold trains on the expanding chronological window (X_train_s_FW,
# y_train_s_FW, both sorted by Date) and evaluates on the following
# wf_horizon_FW rows. ntree = 100 is used in the loop for speed (RF error
# stabilises well before 100 trees); the final model uses ntree = 500.

rf_fold_rmse_FW <- numeric(n_folds_FW)
for (i in seq_len(n_folds_FW)) {
  idx_tr <- wf_slices_FW$train[[i]]
  idx_te <- wf_slices_FW$test[[i]]

  fold_rf_FW      <- randomForest(
    x     = X_train_s_FW[idx_tr, ],
    y     = y_train_s_FW[idx_tr],
    ntree = 100,
    mtry  = floor(ncol(X_train_s_FW) / 3)
  )
  fold_pred_rf_FW   <- predict(fold_rf_FW, newdata = X_train_s_FW[idx_te, ])
  rf_fold_rmse_FW[i] <- Metrics::rmse(y_train_s_FW[idx_te], fold_pred_rf_FW)
  message("[FW] RF fold ", i, "/", n_folds_FW,
          " RMSE: ", round(rf_fold_rmse_FW[i], 4))
}

rf_cv_mean_FW <- round(mean(rf_fold_rmse_FW), 4)
rf_cv_sd_FW   <- round(sd(rf_fold_rmse_FW),   4)
message("[FW] RF walk-forward CV RMSE: ", rf_cv_mean_FW, " \u00b1 ", rf_cv_sd_FW,
        " (", n_folds_FW, " folds)")

# --- Final model (aliased from TS) ------------------------------------------
# TS and FW share the same Date-bounded training set and identical
# hyperparameters; the final fitted model is therefore bit-for-bit identical.
# Aliasing avoids fitting a second ntree = 500 forest in memory while
# rf_model_TS is still resident, which would double RAM usage.
rf_model_FW  <- rf_model_TS
pred_rf_FW   <- pred_rf_TS
rf_imp_df_FW <- rf_imp_df_TS

message("[FW] Random Forest: reuses TS model (same training data and params).")
message("[FW] Random Forest OOB RMSE: ",
        round(sqrt(rf_model_TS$mse[500]), 4))
