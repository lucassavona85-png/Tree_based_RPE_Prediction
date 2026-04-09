# =============================================================================
# 09_figures_FW — ALL FIGURES FOR THE WALK-FORWARD CV (FW) SCHEME
# Project: RPE Prediction in Competitive Runners
# =============================================================================

# All figures follow the project plotting charter.
# theme_project(), save_figure(), prettify_feature(), and project_colors
# are defined in 09_figures_TS.R (sourced first via main.R).

# =============================================================================
# fig_model_comparison_FW — RMSE, MAE, R² bar chart (FW scheme)
# =============================================================================

p_model_comparison_FW <- {
  results_long_FW <- results_FW %>%
    pivot_longer(cols = c(RMSE, MAE, R2),
                 names_to = "Metric", values_to = "Value") %>%
    mutate(
      Model  = factor(Model, levels = c("Dummy Baseline", "Decision Tree",
                                        "Random Forest", "XGBoost")),
      Metric = factor(Metric, levels = c("RMSE", "MAE", "R2"))
    )

  ggplot(results_long_FW,
         aes(x = reorder(Model, Value), y = Value, fill = Model)) +
    geom_col(width = 0.65, show.legend = FALSE) +
    geom_text(aes(label = sprintf("%.3f", Value)),
              hjust = -0.15, size = 3, color = "grey30") +
    facet_wrap(~ Metric, scales = "free_x", nrow = 1) +
    coord_flip() +
    scale_fill_manual(values = project_colors) +
    scale_y_continuous(expand = expansion(mult = c(0, 0.20))) +
    labs(
      title    = "Model comparison \u2014 Walk-Forward CV (FW)",
      subtitle = "Evaluated on chronological held-out test set (last 20 % by Date)",
      x = NULL, y = NULL
    ) +
    theme_project()
}

save_figure(p_model_comparison_FW, "fig_model_comparison_FW", width = 9, height = 3.5)

# =============================================================================
# fig_tree_FW — Decision Tree structure (rpart.plot, base graphics)
# =============================================================================

pdf("figures/fig_tree_FW.pdf", width = 11, height = 7)
rpart.plot(
  dt_model_FW,
  type          = 4,
  extra         = 101,
  fallen.leaves = TRUE,
  shadow.col    = NA,
  branch        = 0.5,
  box.palette   = "Blues",
  main          = "Decision Tree \u2014 Walk-Forward CV (FW)"
)
dev.off()
message("Saved: figures/fig_tree_FW.pdf")

# =============================================================================
# fig_cv_folds_FW — fold-level RMSE per model across the 5 CV folds
# =============================================================================

p_cv_folds_FW <- {
  cv_long <- cv_results_FW %>%
    mutate(
      Model = factor(Model, levels = c("Decision Tree", "Random Forest", "XGBoost")),
      Fold  = as.integer(Fold)
    )

  fold_colors <- c(
    "Decision Tree" = "#457B9D",
    "Random Forest" = "#1D3557",
    "XGBoost"       = "#E63946"
  )

  ggplot(cv_long, aes(x = Fold, y = RMSE, color = Model, group = Model)) +
    geom_line(linewidth = 0.9) +
    geom_point(size = 2.5) +
    scale_color_manual(values = fold_colors) +
    scale_x_continuous(breaks = seq_len(n_folds_FW)) +
    labs(
      title    = "Walk-Forward CV \u2014 RMSE per fold",
      subtitle = paste0(n_folds_FW, " expanding-window folds  |  Assessment horizon = ",
                        wf_horizon_FW, " rows each"),
      x = "Fold", y = "RMSE", color = "Model"
    ) +
    theme_project() +
    theme(legend.position = "top")
}

save_figure(p_cv_folds_FW, "fig_cv_folds_FW", width = 7, height = 4.5)

# =============================================================================
# fig_rf_importance_FW — Random Forest %IncMSE (FW)
# =============================================================================

p_rf_importance_FW <- {
  top15 <- rf_imp_df_FW %>%
    slice(1:min(15, nrow(rf_imp_df_FW))) %>%
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
      title    = "Random Forest \u2014 variable importance (FW)",
      subtitle = "Permutation importance (% increase in MSE on OOB samples)",
      x = NULL, y = "% Increase in MSE"
    ) +
    theme_project()
}

save_figure(p_rf_importance_FW, "fig_rf_importance_FW", width = 7, height = 5)

# =============================================================================
# fig_xgb_importance_FW — XGBoost Gain importance (FW)
# =============================================================================

p_xgb_importance_FW <- {
  top15 <- xgb_imp_df_FW %>%
    slice(1:min(15, nrow(xgb_imp_df_FW))) %>%
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
      title    = "XGBoost \u2014 variable importance (FW)",
      subtitle = "Gain-based importance (average loss reduction per split)",
      x = NULL, y = "Gain"
    ) +
    theme_project()
}

save_figure(p_xgb_importance_FW, "fig_xgb_importance_FW", width = 7, height = 5)

# =============================================================================
# fig_pred_actual_xgb_FW — XGBoost predicted vs actual (FW)
# =============================================================================

p_pred_actual_xgb_FW <- {
  df_plot <- tibble(Actual = y_test_FW, Predicted = pred_xgb_FW)
  ggplot(df_plot, aes(x = Actual, y = Predicted)) +
    geom_point(alpha = 0.25, size = 1.0, color = "#E63946") +
    geom_abline(slope = 1, intercept = 0,
                linetype = "dashed", color = "#1D3557", linewidth = 0.7) +
    labs(
      title    = "XGBoost \u2014 predicted vs actual RPE (FW)",
      subtitle = "Dashed line = perfect prediction  |  RPE on 0\u20131 normalised scale",
      x = "Actual RPE", y = "Predicted RPE"
    ) +
    theme_project()
}

save_figure(p_pred_actual_xgb_FW, "fig_pred_actual_xgb_FW", width = 5, height = 5)

# =============================================================================
# fig_comparison_TS_vs_FW — side-by-side final test RMSE, TS vs FW
# =============================================================================
# Groups bars by model; TS in steel blue (#457B9D), FW in dark navy (#1D3557).
# Excludes Dummy Baseline for readability (it is scheme-agnostic by definition).

p_comparison_TS_vs_FW <- {
  compare_df <- bind_rows(
    results_TS %>% mutate(Scheme = "Temporal Split (TS)"),
    results_FW %>% mutate(Scheme = "Walk-Forward CV (FW)")
  ) %>%
    filter(Model != "Dummy Baseline") %>%
    mutate(
      Model  = factor(Model, levels = c("Decision Tree", "Random Forest", "XGBoost")),
      Scheme = factor(Scheme, levels = c("Temporal Split (TS)", "Walk-Forward CV (FW)"))
    )

  scheme_colors <- c(
    "Temporal Split (TS)"  = "#457B9D",
    "Walk-Forward CV (FW)" = "#1D3557"
  )

  ggplot(compare_df, aes(x = Model, y = RMSE, fill = Scheme)) +
    geom_col(position = position_dodge(width = 0.65), width = 0.55) +
    geom_text(aes(label = sprintf("%.3f", RMSE)),
              position = position_dodge(width = 0.65),
              vjust = -0.4, size = 3, color = "grey30") +
    scale_fill_manual(values = scheme_colors) +
    scale_y_continuous(
      expand = expansion(mult = c(0, 0.12)),
      limits = c(0, NA)
    ) +
    labs(
      title    = "Final test RMSE \u2014 TS vs Walk-Forward CV",
      subtitle = "Same held-out test set (last 20 % by Date) for both schemes",
      x = NULL, y = "RMSE", fill = "Validation scheme"
    ) +
    theme_project() +
    theme(legend.position = "top")
}

save_figure(p_comparison_TS_vs_FW, "fig_comparison_TS_vs_FW", width = 7, height = 4.5)

message("All FW figures exported to figures/")
