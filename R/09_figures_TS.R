# =============================================================================
# 09_figures_TS — ALL FIGURES FOR THE TEMPORAL SPLIT (TS) SCHEME
# Project: RPE Prediction in Competitive Runners
# =============================================================================

# All figures follow the project plotting charter:
# - PDF export via cairo_pdf for lossless LaTeX inclusion
# - PNG export alongside for quick preview
# - rpart.plot exception: uses base graphics via pdf() / dev.off()

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

prettify_feature <- function(x) {
  stringr::str_replace(x, "\\.(\\d)$", " (t-\\1)")
}

# =============================================================================
# fig_model_comparison_TS — RMSE, MAE, R² bar chart (TS scheme)
# =============================================================================

p_model_comparison_TS <- {
  results_long_TS <- results_TS %>%
    pivot_longer(cols = c(RMSE, MAE, R2),
                 names_to = "Metric", values_to = "Value") %>%
    mutate(
      Model  = factor(Model, levels = c("Dummy Baseline", "Decision Tree",
                                        "Random Forest", "XGBoost")),
      Metric = factor(Metric, levels = c("RMSE", "MAE", "R2"))
    )

  ggplot(results_long_TS,
         aes(x = reorder(Model, Value), y = Value, fill = Model)) +
    geom_col(width = 0.65, show.legend = FALSE) +
    geom_text(aes(label = sprintf("%.3f", Value)),
              hjust = -0.15, size = 3, color = "grey30") +
    facet_wrap(~ Metric, scales = "free_x", nrow = 1) +
    coord_flip() +
    scale_fill_manual(values = project_colors) +
    scale_y_continuous(expand = expansion(mult = c(0, 0.20))) +
    labs(
      title    = "Model comparison \u2014 Temporal Split (TS)",
      subtitle = "Evaluated on chronological held-out test set (last 20 % by Date)",
      x = NULL, y = NULL
    ) +
    theme_project()
}

save_figure(p_model_comparison_TS, "fig_model_comparison_TS", width = 9, height = 3.5)

# =============================================================================
# fig_tree_TS — Decision Tree structure (rpart.plot, base graphics)
# =============================================================================

pdf("figures/fig_tree_TS.pdf", width = 11, height = 7)
rpart.plot(
  dt_model_TS,
  type          = 4,
  extra         = 101,
  fallen.leaves = TRUE,
  shadow.col    = NA,
  branch        = 0.5,
  box.palette   = "Blues",
  main          = "Decision Tree \u2014 Temporal Split (TS)"
)
dev.off()
message("Saved: figures/fig_tree_TS.pdf")

# =============================================================================
# fig_rf_importance_TS — Random Forest %IncMSE (TS)
# =============================================================================

p_rf_importance_TS <- {
  top15 <- rf_imp_df_TS %>%
    slice(1:min(15, nrow(rf_imp_df_TS))) %>%
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
      title    = "Random Forest \u2014 variable importance (TS)",
      subtitle = "Permutation importance (% increase in MSE on OOB samples)",
      x = NULL, y = "% Increase in MSE"
    ) +
    theme_project()
}

save_figure(p_rf_importance_TS, "fig_rf_importance_TS", width = 7, height = 5)

# =============================================================================
# fig_xgb_importance_TS — XGBoost Gain importance (TS)
# =============================================================================

p_xgb_importance_TS <- {
  top15 <- xgb_imp_df_TS %>%
    slice(1:min(15, nrow(xgb_imp_df_TS))) %>%
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
      title    = "XGBoost \u2014 variable importance (TS)",
      subtitle = "Gain-based importance (average loss reduction per split)",
      x = NULL, y = "Gain"
    ) +
    theme_project()
}

save_figure(p_xgb_importance_TS, "fig_xgb_importance_TS", width = 7, height = 5)

# =============================================================================
# fig_pred_actual_xgb_TS — XGBoost predicted vs actual (TS)
# =============================================================================

p_pred_actual_xgb_TS <- {
  df_plot <- tibble(Actual = y_test_TS, Predicted = pred_xgb_TS)
  ggplot(df_plot, aes(x = Actual, y = Predicted)) +
    geom_point(alpha = 0.25, size = 1.0, color = "#E63946") +
    geom_abline(slope = 1, intercept = 0,
                linetype = "dashed", color = "#1D3557", linewidth = 0.7) +
    labs(
      title    = "XGBoost \u2014 predicted vs actual RPE (TS)",
      subtitle = "Dashed line = perfect prediction  |  RPE on 0\u20131 normalised scale",
      x = "Actual RPE", y = "Predicted RPE"
    ) +
    theme_project()
}

save_figure(p_pred_actual_xgb_TS, "fig_pred_actual_xgb_TS", width = 5, height = 5)

message("All TS figures exported to figures/")
