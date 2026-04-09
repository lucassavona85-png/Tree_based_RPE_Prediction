# =============================================================================
# 09 — FIGURES
# Project: RPE Prediction in Competitive Runners
# =============================================================================

# All figures follow the project plotting charter:
# - PDF export via cairo_pdf for lossless LaTeX inclusion
# - PNG export for quick preview
# - Consistent palette and theme across all ggplot2 figures
# - rpart.plot exception: uses base graphics via pdf()/dev.off()

dir.create("figures", showWarnings = FALSE)

# --- Project palette and theme -----------------------------------------------

project_colors <- c(
  "Dummy Baseline" = "#CCCCCC",
  "Decision Tree"  = "#457B9D",
  "Random Forest"  = "#1D3557",
  "XGBoost"        = "#E63946"
)

theme_project <- function() {
  theme_minimal(base_size = 11) +
    theme(
      plot.title       = element_text(size = 12, face = "bold",    hjust = 0),
      plot.subtitle    = element_text(size = 10, color = "grey40", hjust = 0),
      plot.caption     = element_text(size =  8, color = "grey60", hjust = 1),
      axis.title       = element_text(size = 10),
      axis.text        = element_text(size =  9, color = "grey30"),
      legend.title     = element_text(size =  9, face = "bold"),
      legend.text      = element_text(size =  9),
      panel.grid.major = element_line(color = "grey92", linewidth = 0.4),
      panel.grid.minor = element_blank(),
      panel.border     = element_blank(),
      axis.line        = element_line(color = "grey70", linewidth = 0.4),
      plot.margin      = ggplot2::margin(12, 12, 8, 12)
    )
}

# --- Export helper -----------------------------------------------------------

save_figure <- function(plot, filename, width = 7, height = 4.5, dpi = 300) {
  ggsave(
    filename = paste0("figures/", filename, ".pdf"),
    plot     = plot,
    width    = width,
    height   = height,
    dpi      = dpi,
    device   = cairo_pdf
  )
  ggsave(
    filename = paste0("figures/", filename, ".png"),
    plot     = plot,
    width    = width,
    height   = height,
    dpi      = dpi
  )
  message("Saved: figures/", filename)
}

# --- Feature name prettifier -------------------------------------------------
# Converts ".1"–".6" suffix to a readable temporal label "(t-1)"–"(t-6)".
# Example: "total km.2" → "total km (t-2)"

prettify_feature <- function(x) {
  stringr::str_replace(x, "\\.(\\d)$", " (t-\\1)")
}

# =============================================================================
# Figure 1 — Model comparison (RMSE, MAE, R²)
# =============================================================================

p_comparison <- {
  results_long <- results %>%
    pivot_longer(cols = c(RMSE, MAE, R2), names_to = "Metric", values_to = "Value") %>%
    mutate(
      Model  = factor(Model, levels = c("Dummy Baseline", "Decision Tree",
                                        "Random Forest", "XGBoost")),
      Metric = factor(Metric, levels = c("RMSE", "MAE", "R2"))
    )

  ggplot(results_long, aes(x = reorder(Model, Value), y = Value, fill = Model)) +
    geom_col(width = 0.65, show.legend = FALSE) +
    geom_text(aes(label = sprintf("%.3f", Value)),
              hjust = -0.15, size = 3, color = "grey30") +
    facet_wrap(~ Metric, scales = "free_x", nrow = 1) +
    coord_flip() +
    scale_fill_manual(values = project_colors) +
    scale_y_continuous(expand = expansion(mult = c(0, 0.20))) +
    labs(
      title    = "Model comparison — RPE prediction",
      subtitle = "Evaluated on chronological held-out test set (20% of data)",
      x = NULL, y = NULL
    ) +
    theme_project()
}

save_figure(p_comparison, "fig_model_comparison", width = 9, height = 3.5)

# =============================================================================
# Figure 2 — Decision Tree plot (rpart.plot — base graphics)
# =============================================================================
# rpart.plot uses base R graphics; exported via pdf()/dev.off() directly.

pdf("figures/fig_tree.pdf", width = 11, height = 7)
rpart.plot(
  dt_model,
  type          = 4,
  extra         = 101,
  fallen.leaves = TRUE,
  shadow.col    = NA,
  branch        = 0.5,
  box.palette   = "Blues",
  main          = "Decision Tree — RPE prediction"
)
dev.off()
message("Saved: figures/fig_tree.pdf")

# =============================================================================
# Figure 3 — Decision Tree variable importance (MDI)
# =============================================================================

p_dt_importance <- {
  top15 <- dt_imp_df %>%
    slice(1:min(15, nrow(dt_imp_df))) %>%
    mutate(
      Feature = prettify_feature(Feature),
      Feature = fct_reorder(Feature, Importance)
    )

  ggplot(top15, aes(x = Feature, y = Importance)) +
    geom_col(fill = "#457B9D", width = 0.7) +
    geom_text(aes(label = round(Importance, 1)),
              hjust = -0.1, size = 3, color = "grey30") +
    coord_flip() +
    scale_y_continuous(expand = expansion(mult = c(0, 0.15))) +
    labs(
      title    = "Decision Tree — variable importance",
      subtitle = "Mean decrease in node impurity (MDI, absolute scale)",
      x = NULL, y = "Impurity decrease"
    ) +
    theme_project()
}

save_figure(p_dt_importance, "fig_dt_importance", width = 7, height = 5)

# =============================================================================
# Figure 4 — Random Forest variable importance (%IncMSE)
# =============================================================================

p_rf_importance <- {
  top15 <- rf_imp_df %>%
    slice(1:min(15, nrow(rf_imp_df))) %>%
    mutate(
      Feature = prettify_feature(Feature),
      Feature = fct_reorder(Feature, `%IncMSE`)
    )

  ggplot(top15, aes(x = Feature, y = `%IncMSE`)) +
    geom_col(fill = "#1D3557", width = 0.7) +
    geom_text(aes(label = round(`%IncMSE`, 1)),
              hjust = -0.1, size = 3, color = "grey30") +
    coord_flip() +
    scale_y_continuous(expand = expansion(mult = c(0, 0.15))) +
    labs(
      title    = "Random Forest — variable importance",
      subtitle = "Permutation importance (% increase in MSE on OOB samples)",
      x = NULL, y = "% Increase in MSE"
    ) +
    theme_project()
}

save_figure(p_rf_importance, "fig_rf_importance", width = 7, height = 5)

# =============================================================================
# Figure 5 — XGBoost variable importance (Gain)
# =============================================================================

p_xgb_importance <- {
  top15 <- xgb_imp_df %>%
    slice(1:min(15, nrow(xgb_imp_df))) %>%
    mutate(
      Feature = prettify_feature(Feature),
      Feature = fct_reorder(Feature, Gain)
    )

  ggplot(top15, aes(x = Feature, y = Gain)) +
    geom_col(fill = "#E63946", width = 0.7) +
    geom_text(aes(label = round(Gain, 3)),
              hjust = -0.1, size = 3, color = "grey30") +
    coord_flip() +
    scale_y_continuous(expand = expansion(mult = c(0, 0.15))) +
    labs(
      title    = "XGBoost — variable importance",
      subtitle = "Gain-based importance (average loss reduction per split)",
      x = NULL, y = "Gain"
    ) +
    theme_project()
}

save_figure(p_xgb_importance, "fig_xgb_importance", width = 7, height = 5)

# =============================================================================
# Figure 6 — Predicted vs Actual (Random Forest)
# =============================================================================

p_pred_rf <- {
  df_plot <- tibble(Actual = y_test, Predicted = rf_pred)
  ggplot(df_plot, aes(x = Actual, y = Predicted)) +
    geom_point(alpha = 0.25, size = 1.0, color = "#1D3557") +
    geom_abline(slope = 1, intercept = 0,
                linetype = "dashed", color = "#E63946", linewidth = 0.7) +
    labs(
      title    = "Random Forest — predicted vs actual RPE",
      subtitle = "Dashed line = perfect prediction  |  RPE on 0–1 normalised scale",
      x = "Actual RPE", y = "Predicted RPE"
    ) +
    theme_project()
}

save_figure(p_pred_rf, "fig_pred_actual_rf", width = 5, height = 5)

# =============================================================================
# Figure 7 — Predicted vs Actual (XGBoost)
# =============================================================================

p_pred_xgb <- {
  df_plot <- tibble(Actual = y_test, Predicted = xgb_pred)
  ggplot(df_plot, aes(x = Actual, y = Predicted)) +
    geom_point(alpha = 0.25, size = 1.0, color = "#E63946") +
    geom_abline(slope = 1, intercept = 0,
                linetype = "dashed", color = "#1D3557", linewidth = 0.7) +
    labs(
      title    = "XGBoost — predicted vs actual RPE",
      subtitle = "Dashed line = perfect prediction  |  RPE on 0–1 normalised scale",
      x = "Actual RPE", y = "Predicted RPE"
    ) +
    theme_project()
}

save_figure(p_pred_xgb, "fig_pred_actual_xgb", width = 5, height = 5)

message("All figures exported to figures/")
