# =============================================================================
# 02 — FEATURE AND TARGET SELECTION
# Project: RPE Prediction in Competitive Runners
# =============================================================================

# --- Select feature and target columns ---------------------------------------
# The dataset is a pre-built sliding window: no lag engineering is needed.
# Feature columns carry suffixes .1 through .6 (days t-1 to t-6).
# Columns with no suffix correspond to day t and MUST be excluded to prevent
# data leakage (they are measured simultaneously with the target).
# `Athlete ID`, `Date`, and `injury` are identifiers/original target — excluded.

feature_cols <- grep("\\.([1-6])$", colnames(df), value = TRUE)

df_model <- df %>%
  select(`Athlete ID`, Date, all_of(feature_cols), `perceived exertion`)

message("Feature columns selected: ", length(feature_cols))
message("Total rows in modelling dataset: ", nrow(df_model))
