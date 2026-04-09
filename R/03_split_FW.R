# =============================================================================
# 03_split_FW — WALK-FORWARD (FW) SCHEME: FOLD DEFINITIONS + SHARED TEST SET
# Project: RPE Prediction in Competitive Runners
# =============================================================================

# --- Shared hold-out test set (identical to TS split) -----------------------
# Both schemes use the same held-out test set so that final metric comparisons
# are not confounded by differences in test-set composition.
# split_date_TS is defined in 03_split_TS.R and must be sourced first.

train_FW <- df_model %>% filter(Date <= split_date_TS)
test_FW  <- df_model %>% filter(Date >  split_date_TS)

X_train_FW <- train_FW %>% select(all_of(feature_cols))
y_train_FW <- train_FW$`perceived exertion`

X_test_FW  <- test_FW  %>% select(all_of(feature_cols))
y_test_FW  <- test_FW$`perceived exertion`

# --- Walk-forward cross-validation slices on the training set ----------------
# Sort training data by global Date before slicing.
# fixedWindow = FALSE: the training window expands at each fold.
# skip = wf_horizon_FW - 1: assessment windows are non-overlapping.
# With initialWindow = 70 % and horizon = 6 %, 5 folds are produced:
#   floor((1 - 0.70) / 0.06) = 5 folds.

train_sorted_FW <- train_FW %>% arrange(Date)

X_train_s_FW <- train_sorted_FW %>% select(all_of(feature_cols))
y_train_s_FW <- train_sorted_FW$`perceived exertion`

n_train_FW    <- nrow(train_sorted_FW)
wf_init_FW    <- floor(n_train_FW * 0.70)   # 70 % initial window
wf_horizon_FW <- floor(n_train_FW * 0.06)   # 6 % horizon per fold

wf_slices_FW <- createTimeSlices(
  y             = seq_len(n_train_FW),
  initialWindow = wf_init_FW,
  horizon       = wf_horizon_FW,
  fixedWindow   = FALSE,
  skip          = wf_horizon_FW - 1   # non-overlapping assessment sets
)

n_folds_FW <- length(wf_slices_FW$train)

# caret trainControl object passed to the Decision Tree tuning step
wf_ctrl_FW <- trainControl(
  method        = "timeslice",
  initialWindow = wf_init_FW,
  horizon       = wf_horizon_FW,
  fixedWindow   = FALSE,
  skip          = wf_horizon_FW - 1
)

message("[FW] Walk-forward CV: ", n_folds_FW, " folds | initialWindow = ",
        wf_init_FW, " | horizon = ", wf_horizon_FW)
message("[FW] Training rows: ", nrow(train_FW), " | Test rows: ", nrow(test_FW))
