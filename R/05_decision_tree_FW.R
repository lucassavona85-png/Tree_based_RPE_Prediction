# =============================================================================
# 05_decision_tree_FW — DECISION TREE: WALK-FORWARD CV (FW) SCHEME
# Project: RPE Prediction in Competitive Runners
# =============================================================================

# --- Prepare syntactically valid column names for caret / rpart -------------
# rpart formulas require syntactically valid R names (no spaces, no hyphens).
# keep a mapping to recover original names when reporting importance.

name_map_dt_FW <- tibble(
  original = feature_cols,
  clean    = make.names(feature_cols)
)

X_train_rp_FW <- X_train_FW %>% rename_with(make.names)
X_test_rp_FW  <- X_test_FW  %>% rename_with(make.names)
target_rp_FW  <- make.names("perceived exertion")

# make.names-renamed sorted training matrix (required by caret train())
X_train_rp_s_FW <- X_train_s_FW %>% rename_with(make.names)

# --- Hyperparameter tuning — complexity parameter (cp) ----------------------
# wf_ctrl_FW (from 03_split_FW.R) applies the walk-forward timeslice CV
# with 5 non-overlapping folds. The fold-level RMSE mean and sd are reported
# as the FW validation metric for model selection.

cp_grid_FW <- expand.grid(cp = seq(0.0001, 0.05, length.out = 30))

set.seed(42)
dt_tune_FW <- train(
  x         = X_train_rp_s_FW,
  y         = y_train_s_FW,
  method    = "rpart",
  tuneGrid  = cp_grid_FW,
  trControl = wf_ctrl_FW,   # walk-forward CV, defined in 03_split_FW.R
  metric    = "RMSE"
)

# Extract mean ± sd RMSE at the best cp from the walk-forward CV folds
best_row_dt_FW  <- which.min(dt_tune_FW$results$RMSE)
dt_cv_mean_FW   <- round(dt_tune_FW$results$RMSE[best_row_dt_FW],   4)
dt_cv_sd_FW     <- round(dt_tune_FW$results$RMSESD[best_row_dt_FW], 4)

message("[FW] DT walk-forward CV RMSE: ", dt_cv_mean_FW, " \u00b1 ", dt_cv_sd_FW,
        " (", n_folds_FW, " folds)")
message("[FW] DT best cp: ", round(dt_tune_FW$bestTune$cp, 6))

# --- Fit final Decision Tree on all FW training data -----------------------
dt_data_FW <- X_train_rp_FW %>%
  mutate(!!target_rp_FW := y_train_FW)

dt_model_FW <- rpart(
  formula = as.formula(paste(target_rp_FW, "~ .")),
  data    = dt_data_FW,
  method  = "anova",
  control = rpart.control(cp = dt_tune_FW$bestTune$cp)
)

# --- Predict on held-out test set --------------------------------------------
pred_dt_FW <- predict(dt_model_FW, newdata = X_test_rp_FW)

# --- Variable importance (MDI) -----------------------------------------------
dt_imp_df_FW <- tibble(
  clean      = names(dt_model_FW$variable.importance),
  Importance = dt_model_FW$variable.importance
) %>%
  left_join(name_map_dt_FW, by = "clean") %>%
  select(Feature = original, Importance) %>%
  arrange(desc(Importance))

# Per-fold DT RMSE for cv_results_FW — caret's $resample has one row per fold
# (at the selected best cp). Each row corresponds to one walk-forward fold in
# the same order as wf_slices_FW, so we extract the RMSE column directly.
dt_fold_rmse_FW <- dt_tune_FW$resample$RMSE

message("[FW] Decision Tree fitted. Leaves: ",
        sum(dt_model_FW$frame$var == "<leaf>"))
