# ==============================================================================
# 07_shap_analysis.R
# SHAP interpretation of the locked final CatBoost model
# ==============================================================================

library(catboost)
library(shapviz)
library(dplyr)
library(ggplot2)
library(patchwork)

source("RR/main/00_setup.R")


# ==============================================================================
# 1. Paths
# ==============================================================================

locked_catboost_file <- file.path(
  model_dir,
  "locked",
  "model_cat.rds"
)

train_file <- file.path(
  processed_dir,
  "train_set.rds"
)


if (!file.exists(locked_catboost_file)) {
  stop(
    "Locked CatBoost model was not found: ",
    locked_catboost_file
  )
}

if (!file.exists(train_file)) {
  stop(
    "Training dataset was not found: ",
    train_file
  )
}
# ==============================================================================
# 2. Load locked model and training data
# ==============================================================================

model_cat <- readRDS(
  locked_catboost_file
)

train_set <- readRDS(
  train_file
)


stopifnot(
  nrow(train_set) == 6907,
  sum(train_set$met_diagnosis == "1") == 1538,
  all(complete.cases(train_set))
)


X_train <- train_set[
  ,
  final_predictors
]


stopifnot(
  identical(
    colnames(X_train),
    final_predictors
  )
)
# ==============================================================================
# 3. Variable labels
# ==============================================================================

label_map <- c(
  "BMI"   = "BMI (kg/m2)",
  "HbA1c" = "HbA1c (%)",
  "UA"    = "UA (mg/dL)",
  "pulse" = "Pulse (beats/min)",
  "age"   = "Age (years)",
  "TC"    = "TC (mg/dL)",
  "BUN"   = "BUN (mg/dL)"
)
# ==============================================================================
# 4. Calculate native CatBoost SHAP values
# ==============================================================================

cat(
  "Calculating SHAP values from the locked CatBoost model...\n"
)


pool_shap <- catboost.load_pool(
  data = X_train
)


shap_contribution <- catboost.get_feature_importance(
  model_cat,
  pool = pool_shap,
  type = "ShapValues"
)


stopifnot(
  nrow(shap_contribution) == nrow(X_train),
  ncol(shap_contribution) ==
    length(final_predictors) + 1
)
# The final column returned by CatBoost ShapValues
# is the expected value / baseline.
shap_matrix <- shap_contribution[
  ,
  -ncol(shap_contribution),
  drop = FALSE
]


colnames(shap_matrix) <-
  final_predictors


stopifnot(
  ncol(shap_matrix) == 7,
  identical(
    colnames(shap_matrix),
    final_predictors
  )
)
# ==============================================================================
# 5. Save SHAP values
# ==============================================================================

saveRDS(
  shap_matrix,
  file.path(
    results_dir,
    "CatBoost_SHAP_values.rds"
  )
)


write.csv(
  shap_matrix,
  file.path(
    results_dir,
    "CatBoost_SHAP_values.csv"
  ),
  row.names = FALSE
)
# ==============================================================================
# 6. Create shapviz object
# ==============================================================================

shp <- shapviz(
  shap_matrix,
  X_train
)


shp_labeled <- shp

colnames(
  shp_labeled$S
) <- label_map[
  colnames(shp_labeled$S)
]

colnames(
  shp_labeled$X
) <- label_map[
  colnames(shp_labeled$X)
]
# ==============================================================================
# 7. Global SHAP importance
# ==============================================================================

mean_abs_shap <- colMeans(
  abs(
    as.matrix(
      shap_matrix
    )
  )
)


shap_importance <- data.frame(
  Feature =
    names(mean_abs_shap),
  
  Feature_with_Unit =
    unname(
      label_map[
        names(mean_abs_shap)
      ]
    ),
  
  Mean_Abs_SHAP =
    as.numeric(
      mean_abs_shap
    )
) %>%
  arrange(
    desc(Mean_Abs_SHAP)
  )


write.csv(
  shap_importance,
  file.path(
    results_dir,
    "SHAP_global_importance.csv"
  ),
  row.names = FALSE
)
# ==============================================================================
# 8. Identify the four most important predictors
# ==============================================================================

top_vars <- shap_importance$Feature[
  1:4
]


stopifnot(
  setequal(
    top_vars,
    c(
      "BMI",
      "HbA1c",
      "UA",
      "pulse"
    )
  )
)
# ==============================================================================
# 9. SHAP summary plot
# ==============================================================================

pdf(
  file.path(
    results_dir,
    "Figure6_SHAP_summary.pdf"
  ),
  width = 11,
  height = 7
)


p_summary <- sv_importance(
  shp_labeled,
  kind = "beeswarm"
) +
  theme_bw() +
  theme(
    axis.text =
      element_text(
        color = "black",
        size = 10
      ),
    axis.title =
      element_text(
        size = 12,
        face = "bold"
      )
  )


print(
  p_summary
)


dev.off()
# ==============================================================================
# 10. SHAP global-importance bar plot
# ==============================================================================

pdf(
  file.path(
    results_dir,
    "SHAP_importance_bar.pdf"
  ),
  width = 9,
  height = 6
)


p_importance <- sv_importance(
  shp_labeled,
  kind = "bar"
) +
  theme_bw() +
  theme(
    axis.text =
      element_text(
        color = "black",
        size = 10
      ),
    axis.title =
      element_text(
        size = 12,
        face = "bold"
      )
  )


print(
  p_importance
)


dev.off()
# ==============================================================================
# 11. SHAP dependence plots for the top four predictors
# ==============================================================================

plot_list <- list()


for (v in top_vars) {
  
  x_limits <- quantile(
    X_train[[v]],
    probs = c(
      0.025,
      0.975
    ),
    na.rm = TRUE
  )
  
  
  df_plot <- data.frame(
    x = X_train[[v]],
    shap = shap_matrix[, v]
  )
  
  
  df_smooth <- df_plot %>%
    filter(
      x >= x_limits[1],
      x <= x_limits[2]
    )
  
  
  p <- sv_dependence(
    shp,
    v,
    alpha = 0.3,
    size = 1.2,
    color_var = NULL
  ) +
    
    geom_smooth(
      data = df_smooth,
      aes(
        x = x,
        y = shap
      ),
      method = "loess",
      se = FALSE,
      span = 0.8
    ) +
    
    geom_rug(
      data = df_plot,
      aes(
        x = x
      ),
      sides = "b",
      alpha = 0.1
    ) +
    
    scale_y_continuous(
      expand = expansion(
        mult = c(
          0.10,
          0.15
        )
      )
    ) +
    
    labs(
      x = label_map[v],
      y =
        "SHAP value (log-odds scale)",
      title =
        paste(
          "Impact of",
          v
        )
    ) +
    
    theme_bw() +
    
    theme(
      plot.title =
        element_text(
          size = 12,
          face = "bold",
          hjust = 0.5
        ),
      axis.title =
        element_text(
          size = 11,
          color = "black"
        ),
      axis.text =
        element_text(
          size = 10,
          color = "black"
        ),
      panel.grid.minor =
        element_blank()
    )
  
  
  plot_list[[v]] <- p
}
combined_dependence <- wrap_plots(
  plot_list,
  ncol = 2,
  nrow = 2
)


pdf(
  file.path(
    results_dir,
    "Figure7_SHAP_dependence_top4.pdf"
  ),
  width = 14,
  height = 11
)


print(
  combined_dependence
)


dev.off()
# ==============================================================================
# 12. Save top-four feature/SHAP pairs for reproducibility
# ==============================================================================

top4_shap_data <- data.frame(
  BMI =
    X_train$BMI,
  BMI_SHAP =
    shap_matrix[, "BMI"],
  
  HbA1c =
    X_train$HbA1c,
  HbA1c_SHAP =
    shap_matrix[, "HbA1c"],
  
  UA =
    X_train$UA,
  UA_SHAP =
    shap_matrix[, "UA"],
  
  pulse =
    X_train$pulse,
  pulse_SHAP =
    shap_matrix[, "pulse"]
)


saveRDS(
  top4_shap_data,
  file.path(
    results_dir,
    "SHAP_top4_feature_pairs.rds"
  )
)
# ==============================================================================
# 13. Reproducibility summary
# ==============================================================================

cat("\n")
cat("============================================================\n")
cat("SHAP reproducibility summary\n")
cat("============================================================\n")


print(
  shap_importance
)


cat(
  "\nTop four predictors:",
  paste(
    top_vars,
    collapse = ", "
  ),
  "\n"
)


cat(
  "BMI / HbA1c importance ratio =",
  mean_abs_shap["BMI"] /
    mean_abs_shap["HbA1c"],
  "\n"
)


cat("============================================================\n")