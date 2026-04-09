# =============================================================================
# 02b — EXPLORATORY DATA ANALYSIS FIGURES
# Project: RPE Prediction in Competitive Runners
# =============================================================================
# Source this script AFTER 03_split_TS.R (needs train_TS, feature_cols, df_model).
# Produces two EDA figures for Section 3.4 of the article.

dir.create("figures", showWarnings = FALSE)

# --- Project palette and theme (same as 09_figures_TS.R) --------------------

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
    plot     = plot, width = width, height = height, dpi = dpi,
    device   = cairo_pdf
  )
  ggsave(
    filename = paste0("figures/", filename, ".png"),
    plot     = plot, width = width, height = height, dpi = dpi
  )
  message("Saved: figures/", filename)
}

# =============================================================================
# Figure A — RPE histogram on training set with mean and median lines
# =============================================================================

p_rpe_hist <- {
  rpe_mean   <- mean(y_train_TS)
  rpe_median <- median(y_train_TS)

  ggplot(tibble(RPE = y_train_TS), aes(x = RPE)) +
    geom_histogram(bins = 40, fill = "#457B9D", color = "white",
                   linewidth = 0.2, alpha = 0.85) +
    geom_vline(aes(xintercept = rpe_mean),
               linetype = "dashed", color = "#E63946", linewidth = 0.7) +
    geom_vline(aes(xintercept = rpe_median),
               linetype = "dotted", color = "#1D3557", linewidth = 0.7) +
    annotate("text", x = rpe_mean + 0.03, y = Inf, vjust = 1.5,
             label = paste0("Mean = ", round(rpe_mean, 3)),
             color = "#E63946", size = 3.2, hjust = 0) +
    annotate("text", x = rpe_median - 0.03, y = Inf, vjust = 3,
             label = paste0("Median = ", round(rpe_median, 3)),
             color = "#1D3557", size = 3.2, hjust = 1) +
    labs(
      title    = "Distribution of RPE on the training set",
      subtitle = "Normalised RPE (0\u20131 scale)  |  Chronological 80/20 split",
      x = "RPE (normalised)", y = "Count"
    ) +
    theme_project()
}

save_figure(p_rpe_hist, "fig_rpe_histogram", width = 5, height = 3.5)

# =============================================================================
# Figure B — Correlation heatmap of the 10 variables at lag t-1
# =============================================================================

p_corr_heatmap <- {
  # Select only .1 suffix columns (lag t-1)
  lag1_cols <- grep("\\.1$", feature_cols, value = TRUE)

  cor_mat <- cor(train_TS %>% select(all_of(lag1_cols)), use = "pairwise.complete.obs")

  # Pretty labels: remove the ".1" suffix
  pretty_names <- stringr::str_remove(colnames(cor_mat), "\\.1$")
  colnames(cor_mat) <- pretty_names
  rownames(cor_mat) <- pretty_names

  # Convert to long format for ggplot
  cor_long <- as.data.frame(as.table(cor_mat)) %>%
    rename(Var1 = 1, Var2 = 2, Correlation = 3)

  ggplot(cor_long, aes(x = Var1, y = Var2, fill = Correlation)) +
    geom_tile(color = "white", linewidth = 0.3) +
    geom_text(aes(label = sprintf("%.2f", Correlation)),
              size = 2.2, color = "black") +
    scale_fill_gradient2(low = "#457B9D", mid = "white", high = "#E63946",
                         midpoint = 0, limits = c(-1, 1)) +
    labs(
      title    = "Pairwise correlation of features at lag t\u22121",
      subtitle = "Computed on the training set (34,226 observations)",
      x = NULL, y = NULL
    ) +
    theme_project() +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1, size = 7),
      axis.text.y = element_text(size = 7),
      legend.position = "right"
    )
}

save_figure(p_corr_heatmap, "fig_correlation_heatmap", width = 5.5, height = 4.5)

message("EDA figures exported to figures/")
