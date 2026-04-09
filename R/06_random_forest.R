# =============================================================================
# 06 — RANDOM FOREST
# Project: RPE Prediction in Competitive Runners
# =============================================================================

# --- Walk-forward cross-validation for model selection reporting -------------
# Each fold trains on the expanding chronological window defined in 03_split.R
# (X_train_s / y_train_s, sorted by Date) and evaluates on the next wf_horizon
# rows. ntree = 100 is used in the fold loop for speed; RF error stabilises
# well before 100 trees, so this gives a reliable RMSE estimate. The final
# model below uses ntree = 500 for maximum predictive accuracy.

rf_fold_rmse <- numeric(n_folds)
for (i in seq_len(n_folds)) {
  idx_tr <- wf_slices$train[[i]]
  idx_te <- wf_slices$test[[i]]

  fold_rf         <- randomForest(x     = X_train_s[idx_tr, ],
                                   y     = y_train_s[idx_tr],
                                   ntree = 100,
                                   mtry  = floor(ncol(X_train_s) / 3))
  fold_pred        <- predict(fold_rf, newdata = X_train_s[idx_te, ])
  rf_fold_rmse[i]  <- Metrics::rmse(y_train_s[idx_te], fold_pred)
  message("RF fold ", i, "/", n_folds, " RMSE: ", round(rf_fold_rmse[i], 4))
}

rf_cv_mean <- round(mean(rf_fold_rmse), 4)
rf_cv_sd   <- round(sd(rf_fold_rmse),   4)
message("RF walk-forward CV RMSE: ", rf_cv_mean, " \u00b1 ", rf_cv_sd,
        " (", n_folds, " folds)")

# --- Fit Random Forest -------------------------------------------------------
# randomForest uses the data-frame / matrix interface directly (no formula),
# so column names with special characters are handled without transformation.
#
# ntree = 500: sufficient for stable OOB error and importance estimates.
# mtry = floor(p / 3): default for regression; samples p/3 candidates per split.
# importance = TRUE: required to extract both %IncMSE and IncNodePurity later.
#
# Tree-based models require no feature normalisation: the recursive split
# criterion is invariant to monotone transformations of feature values.

set.seed(42)
rf_model <- randomForest(
  x          = X_train,
  y          = y_train,
  ntree      = 500,
  mtry       = floor(ncol(X_train) / 3),
  importance = TRUE
)

# --- Predict on test set -----------------------------------------------------
rf_pred <- predict(rf_model, newdata = X_test)

# --- Variable importance (permutation — %IncMSE) ----------------------------
# Permutation importance (%IncMSE): average increase in prediction error when
# a feature's values are randomly permuted across out-of-bag samples.
# Preferred over IncNodePurity (MDI) which is biased toward high-cardinality
# features. Permutation importance is more reliable for interpretation.

rf_imp_df <- importance(rf_model) %>%
  as.data.frame() %>%
  rownames_to_column("Feature") %>%
  arrange(desc(`%IncMSE`))

message("Random Forest OOB RMSE: ", round(sqrt(rf_model$mse[500]), 4))
