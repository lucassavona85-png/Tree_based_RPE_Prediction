# =============================================================================
# 01 — DATA LOADING AND CLEANING
# Project: RPE Prediction in Competitive Runners
# =============================================================================

# --- Load raw dataset --------------------------------------------------------
# The CSV is a pre-built sliding window: each row is one prediction instance.
# Columns with no suffix = day t (current day). Columns with .1–.6 = days t-1 to t-6.

df_raw <- read_csv(
  "day_approach_maskedID_timeseries.csv",
  show_col_types = FALSE
)

# --- Filter and sort ---------------------------------------------------------
# Never impute the target: rows with missing perceived exertion are removed.
# Sorting by Athlete ID then Date ensures athletes are processed in time order.

df <- df_raw %>%
  filter(!is.na(`perceived exertion`)) %>%
  arrange(`Athlete ID`, Date)

message("Rows after removing missing target: ", nrow(df),
        " (removed ", nrow(df_raw) - nrow(df), ")")
