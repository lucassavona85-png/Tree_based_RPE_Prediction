# =============================================================================
# 05 — DECISION TREE
# Project: RPE Prediction in Competitive Runners
# =============================================================================

# --- Prepare syntactically valid column names for caret / rpart -------------
# rpart formulas and caret's model.frame require syntactically valid R names.
# We rename columns with make.names(), store the mapping, and reverse it after
# fitting to recover original feature names for importance reporting.

name_map_dt <- tibble(
  original = feature_cols,
  clean    = make.names(feature_cols)
)

X_train_rp <- X_train %>% rename_with(make.names)
X_test_rp  <- X_test  %>% rename_with(make.names)
target_rp  <- make.names("perceived exertion")   # "perceived.exertion"

# --- Hyperparameter tuning — complexity parameter (cp) ----------------------
# cp is the key hyperparameter for CART: a split is only made if it reduces
# impurity by at least cp × root-node impurity. Larger cp = simpler tree.
# wf_ctrl (from 03_split.R) applies walk-forward CV with ~5 non-overlapping
# folds on the chronologically sorted training data.

# X_train_rp_s: make.names-renamed version of X_train_s (sorted by Date,
# defined in 03_split.R) — required because rpart formulas reject names with
# spaces; train_sorted, y_train_s, and wf_ctrl are all from 03_split.R.
X_train_rp_s <- X_train_s %>% rename_with(make.names)

cp_grid <- expand.grid(cp = seq(0.0001, 0.05, length.out = 30))

set.seed(42)
dt_tune <- train(
  x         = X_train_rp_s,
  y         = y_train_s,
  method    = "rpart",
  tuneGrid  = cp_grid,
  trControl = wf_ctrl,          # walk-forward CV, defined in 03_split.R
  metric    = "RMSE"
)

# dt_tune$results: one row per cp value; RMSE is the mean across CV folds,
# RMSESD is the standard deviation — both from the walk-forward timeslice CV.
best_row   <- which.min(dt_tune$results$RMSE)
dt_cv_mean <- round(dt_tune$results$RMSE[best_row],   4)
dt_cv_sd   <- round(dt_tune$results$RMSESD[best_row], 4)
message("DT walk-forward CV RMSE: ", dt_cv_mean, " \u00b1 ", dt_cv_sd,
        " (", n_folds, " folds)")
message("Best cp: ", round(dt_tune$bestTune$cp, 6))

# --- Fit final Decision Tree on all training data ----------------------------
# The model is refitted on the full training set using the optimal cp found
# above. method = "anova" specifies regression (sum-of-squares splitting).

dt_data <- X_train_rp %>%
  mutate(!!target_rp := y_train)

dt_model <- rpart(
  formula = as.formula(paste(target_rp, "~ .")),
  data    = dt_data,
  method  = "anova",
  control = rpart.control(cp = dt_tune$bestTune$cp)
)

# --- Predict on test set -----------------------------------------------------
dt_pred <- predict(dt_model, newdata = X_test_rp)

# --- Variable importance (MDI) -----------------------------------------------
# Native importance from rpart: accumulated reduction in node impurity (MSE)
# attributed to each feature across all splits. Expressed in absolute scale.

dt_imp_df <- tibble(
  clean      = names(dt_model$variable.importance),
  Importance = dt_model$variable.importance
) %>%
  left_join(name_map_dt, by = "clean") %>%
  select(Feature = original, Importance) %>%
  arrange(desc(Importance))

message("Decision Tree fitted. Leaves: ", sum(dt_model$frame$var == "<leaf>"))
