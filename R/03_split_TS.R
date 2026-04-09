# =============================================================================
# 03_split_TS — TEMPORAL SPLIT (TS) SCHEME: TRAIN / TEST
# Project: RPE Prediction in Competitive Runners
# =============================================================================

# --- Chronological hold-out split on the global Date index ------------------
# Date is an integer (0–2673) shared across all athletes: it represents a
# global time axis. Splitting at the 80th percentile of Date ensures the test
# set contains only observations that follow all training observations in time.
# No cross-validation is used; the model is fitted once on train and evaluated
# once on test.

split_date_TS <- quantile(df_model$Date, 0.80)

train_TS <- df_model %>% filter(Date <= split_date_TS)
test_TS  <- df_model %>% filter(Date >  split_date_TS)

X_train_TS <- train_TS %>% select(all_of(feature_cols))
y_train_TS <- train_TS$`perceived exertion`

X_test_TS  <- test_TS  %>% select(all_of(feature_cols))
y_test_TS  <- test_TS$`perceived exertion`

message("[TS] Split date (80th pct): ", split_date_TS)
message("[TS] Training rows: ", nrow(train_TS), " | Test rows: ", nrow(test_TS))
