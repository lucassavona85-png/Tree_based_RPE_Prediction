# =============================================================================
# 06_random_forest_TS — RANDOM FOREST: TEMPORAL SPLIT (TS) SCHEME
# Project: RPE Prediction in Competitive Runners
# =============================================================================

# --- Fit Random Forest on the TS training set --------------------------------
# randomForest accepts the data-frame / matrix interface directly, so column
# names with special characters are handled without transformation.
# ntree = 500: sufficient for stable OOB error and importance estimates.
# mtry = floor(p / 3): default for regression.
# importance = TRUE: required to extract permutation importance (%IncMSE).
# Tree-based models require no feature normalisation: the split criterion is
# invariant to monotone transformations of feature values.

set.seed(42)
rf_model_TS <- randomForest(
  x          = X_train_TS,
  y          = y_train_TS,
  ntree      = 500,
  mtry       = floor(ncol(X_train_TS) / 3),
  importance = TRUE
)

# --- Predict on held-out test set --------------------------------------------
pred_rf_TS <- predict(rf_model_TS, newdata = X_test_TS)

# --- Variable importance (permutation — %IncMSE) ----------------------------
# %IncMSE: average increase in prediction error when a feature's values are
# randomly permuted across out-of-bag samples. More reliable than IncNodePurity
# (MDI), which is biased toward high-cardinality features.

rf_imp_df_TS <- importance(rf_model_TS) %>%
  as.data.frame() %>%
  rownames_to_column("Feature") %>%
  arrange(desc(`%IncMSE`))

message("[TS] Random Forest OOB RMSE: ",
        round(sqrt(rf_model_TS$mse[500]), 4))
