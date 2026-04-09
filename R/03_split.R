# =============================================================================
# 03 — CHRONOLOGICAL TRAIN / TEST SPLIT + WALK-FORWARD CV SLICES
# Project: RPE Prediction in Competitive Runners
# =============================================================================

# --- Chronological hold-out split on the global Date index ------------------
# Date is an integer (0–2673) shared across all athletes: it represents a
# global time axis. Splitting at the 80th percentile of Date ensures the test
# set contains only observations that follow all training observations in time.
# The held-out test set is never used during cross-validation or model fitting.

split_date <- quantile(df_model$Date, 0.80)

train <- df_model %>% filter(Date <= split_date)
test  <- df_model %>% filter(Date >  split_date)

X_train <- train %>% select(all_of(feature_cols))
y_train <- train$`perceived exertion`

X_test  <- test  %>% select(all_of(feature_cols))
y_test  <- test$`perceived exertion`

message("Hold-out split date (80th pct): ", split_date)
message("Training rows: ", nrow(train), " | Test rows: ", nrow(test))

# --- Walk-forward cross-validation slices on the training set ----------------
# Sort training data by global Date before slicing; within-day athlete order
# is arbitrary and does not carry temporal information.
# fixedWindow = FALSE: the training window grows at each fold (expanding window).
# skip = wf_horizon - 1: assessment windows are non-overlapping.
# With initialWindow = 70 % and horizon = 6 % of training rows,
# floor((1 − 0.70) / 0.06) = 5 folds are produced.

train_sorted <- train %>% arrange(Date)

X_train_s <- train_sorted %>% select(all_of(feature_cols))
y_train_s <- train_sorted$`perceived exertion`

n_train    <- nrow(train_sorted)
wf_init    <- floor(n_train * 0.70)   # expanding initial window  (70 %)
wf_horizon <- floor(n_train * 0.06)   # assessment horizon per fold (6 %)

wf_slices <- createTimeSlices(
  y             = seq_len(n_train),
  initialWindow = wf_init,
  horizon       = wf_horizon,
  fixedWindow   = FALSE,
  skip          = wf_horizon - 1       # non-overlapping folds
)

n_folds <- length(wf_slices$train)
message("Walk-forward CV: ", n_folds, " folds | initialWindow = ", wf_init,
        " | horizon = ", wf_horizon)

# caret trainControl object — used by the Decision Tree hyperparameter tuning
wf_ctrl <- trainControl(
  method        = "timeslice",
  initialWindow = wf_init,
  horizon       = wf_horizon,
  fixedWindow   = FALSE,
  skip          = wf_horizon - 1
)
