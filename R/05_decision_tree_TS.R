# =============================================================================
# 05_decision_tree_TS — DECISION TREE: TEMPORAL SPLIT (TS) SCHEME
# Project: RPE Prediction in Competitive Runners
# =============================================================================

# --- Prepare syntactically valid column names for caret / rpart -------------
# rpart formulas and caret's model.frame require syntactically valid R names.
# We rename with make.names(), store the mapping, and recover original names
# for importance reporting.

name_map_dt_TS <- tibble(
  original = feature_cols,
  clean    = make.names(feature_cols)
)

X_train_rp_TS <- X_train_TS %>% rename_with(make.names)
X_test_rp_TS  <- X_test_TS  %>% rename_with(make.names)
target_rp_TS  <- make.names("perceived exertion")   # "perceived.exertion"

# --- Hyperparameter tuning — complexity parameter (cp) ----------------------
# cp is the key hyperparameter for CART: a split is only made if it reduces
# impurity by at least cp × root-node impurity. Larger cp = simpler tree.
# Under the TS scheme, a lightweight internal timeslice CV is used only to
# select cp; it is not reported as the validation metric.

n_train_TS_inner <- nrow(train_TS)
ts_ctrl_TS <- trainControl(
  method        = "timeslice",
  initialWindow = floor(n_train_TS_inner * 0.70),
  horizon       = floor(n_train_TS_inner * 0.10),
  fixedWindow   = FALSE,
  skip          = floor(n_train_TS_inner * 0.10) - 1
)

X_train_rp_sorted_TS <- train_TS %>%
  arrange(Date) %>%
  select(all_of(feature_cols)) %>%
  rename_with(make.names)

y_train_sorted_TS <- train_TS %>% arrange(Date) %>% pull(`perceived exertion`)

cp_grid_TS <- expand.grid(cp = seq(0.0001, 0.05, length.out = 30))

set.seed(42)
dt_tune_TS <- train(
  x         = X_train_rp_sorted_TS,
  y         = y_train_sorted_TS,
  method    = "rpart",
  tuneGrid  = cp_grid_TS,
  trControl = ts_ctrl_TS,
  metric    = "RMSE"
)

message("[TS] DT best cp: ", round(dt_tune_TS$bestTune$cp, 6))

# --- Fit final Decision Tree on the TS training set --------------------------
# Refitted on the full TS training split using the optimal cp found above.

dt_data_TS <- X_train_rp_TS %>%
  mutate(!!target_rp_TS := y_train_TS)

dt_model_TS <- rpart(
  formula = as.formula(paste(target_rp_TS, "~ .")),
  data    = dt_data_TS,
  method  = "anova",
  control = rpart.control(cp = dt_tune_TS$bestTune$cp)
)

# --- Predict on held-out test set --------------------------------------------
pred_dt_TS <- predict(dt_model_TS, newdata = X_test_rp_TS)

# --- Variable importance (MDI) -----------------------------------------------
dt_imp_df_TS <- tibble(
  clean      = names(dt_model_TS$variable.importance),
  Importance = dt_model_TS$variable.importance
) %>%
  left_join(name_map_dt_TS, by = "clean") %>%
  select(Feature = original, Importance) %>%
  arrange(desc(Importance))

message("[TS] Decision Tree fitted. Leaves: ",
        sum(dt_model_TS$frame$var == "<leaf>"))
