# =============================================================================
# 04 — DUMMY BASELINE
# Project: RPE Prediction in Competitive Runners
# =============================================================================

# --- Constant-mean predictor -------------------------------------------------
# The dummy baseline predicts the training-set mean for every test observation.
# Any model that does not outperform this baseline has learned nothing from
# the feature set. It provides a lower bound for the comparison table.

dummy_pred <- rep(mean(y_train), nrow(test))

message("Dummy baseline value (training mean): ", round(mean(y_train), 4))
