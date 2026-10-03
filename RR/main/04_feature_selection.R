# ==============================================================================
# 04_feature_selection.R
# LASSO variable selection and multicollinearity assessment
# ==============================================================================

library(glmnet)
library(dplyr)
library(car)
library(corrplot)

source("RR/main/00_setup.R")


# ==============================================================================
# 1. Load the training dataset
# ==============================================================================

train_file <- file.path(
  processed_dir,
  "train_set.rds"
)

if (!file.exists(train_file)) {
  stop(
    "Training dataset was not found. ",
    "Run RR/main/03_descriptive_analysis_and_split.R first."
  )
}

train_set <- readRDS(train_file)


# Reproducibility checks for the training cohort
stopifnot(
  nrow(train_set) == 6907,
  sum(train_set$met_diagnosis == "1") == 1538,
  all(complete.cases(train_set))
)
# ==============================================================================
# 2. Prepare the 14 predictors entering LASSO
# ==============================================================================

stopifnot(
  length(lasso_candidate_variables) == 14,
  all(
    lasso_candidate_variables %in%
      names(train_set)
  )
)

x <- as.matrix(
  train_set[, lasso_candidate_variables]
)

y <- as.numeric(
  as.character(
    train_set$met_diagnosis
  )
)
# ==============================================================================
# 3. LASSO fitting and 10-fold cross-validation
# ==============================================================================

set.seed(SEED)

lasso_fit <- glmnet(
  x = x,
  y = y,
  family = "binomial",
  alpha = 1,
  standardize = TRUE
)

cv_lasso <- cv.glmnet(
  x = x,
  y = y,
  family = "binomial",
  alpha = 1,
  nfolds = 10,
  standardize = TRUE
)
# ==============================================================================
# 4. Extract predictors selected by the 1-SE rule
# ==============================================================================

coefficients_1se <- coef(
  cv_lasso,
  s = "lambda.1se"
)

lasso_selected <- data.frame(
  Predictor = rownames(coefficients_1se),
  Coefficient = as.numeric(coefficients_1se)
) %>%
  filter(
    Predictor != "(Intercept)",
    Coefficient != 0
  )
expected_lasso_variables <- c(
  "age",
  "BMI",
  "pulse",
  "BUN",
  "LDL_C",
  "TC",
  "UA",
  "HbA1c"
)
# ==============================================================================
# 5. Verify LASSO selection against the reported analysis
# ==============================================================================

stopifnot(
  nrow(lasso_selected) == 8,
  setequal(
    lasso_selected$Predictor,
    expected_lasso_variables
  )
)

lambda_1se <- cv_lasso$lambda.1se
lambda_min <- cv_lasso$lambda.min
# ==============================================================================
# 6. Save LASSO results
# ==============================================================================

write.csv(
  lasso_selected,
  file.path(
    results_dir,
    "LASSO_selected_features.csv"
  ),
  row.names = FALSE
)


lambda_summary <- data.frame(
  lambda_min = lambda_min,
  lambda_1se = lambda_1se,
  n_selected_1se = nrow(lasso_selected)
)

write.csv(
  lambda_summary,
  file.path(
    results_dir,
    "LASSO_lambda_summary.csv"
  ),
  row.names = FALSE
)
# ==============================================================================
# 7. Save LASSO coefficient-path and cross-validation plots
# ==============================================================================

pdf(
  file.path(
    results_dir,
    "LASSO_selection_plots.pdf"
  ),
  width = 12,
  height = 6
)

par(
  mfrow = c(1, 2)
)

plot(
  lasso_fit,
  xvar = "lambda",
  label = TRUE
)

title(
  "A: LASSO Coefficient Paths",
  line = 2.5
)

plot(
  cv_lasso
)

abline(
  v = log(cv_lasso$lambda.1se),
  lty = 2
)

title(
  "B: 10-fold Cross-validation",
  line = 2.5
)

dev.off()
# ==============================================================================
# 8. Multicollinearity assessment among LASSO-selected predictors
# ==============================================================================

lasso_vars <- expected_lasso_variables

vif_model <- glm(
  met_diagnosis ~ .,
  data = train_set[
    ,
    c(
      lasso_vars,
      "met_diagnosis"
    )
  ],
  family = binomial()
)

vif_values <- car::vif(
  vif_model
)


vif_results <- data.frame(
  Predictor = names(vif_values),
  VIF = as.numeric(vif_values)
)
write.csv(
  vif_results,
  file.path(
    results_dir,
    "VIF_results.csv"
  ),
  row.names = FALSE
)
# ==============================================================================
# 9. Correlation analysis among LASSO-selected predictors
# ==============================================================================

correlation_matrix <- cor(
  train_set[, lasso_vars],
  method = "pearson"
)

tc_ldl_correlation <- correlation_matrix[
  "TC",
  "LDL_C"
]
write.csv(
  correlation_matrix,
  file.path(
    results_dir,
    "LASSO_selected_feature_correlations.csv"
  ),
  row.names = TRUE
)
# ==============================================================================
# 10. Save correlation heatmap
# ==============================================================================

pdf(
  file.path(
    results_dir,
    "LASSO_selected_feature_correlation_heatmap.pdf"
  ),
  width = 8,
  height = 8
)

corrplot(
  correlation_matrix,
  method = "color",
  addCoef.col = "black",
  tl.col = "black",
  tl.srt = 45,
  title = "\n\nCorrelation Matrix of LASSO-Selected Features",
  mar = c(0, 0, 1, 0)
)

dev.off()
# ==============================================================================
# 11. Define the final predictor set
# ==============================================================================

final_selected_predictors <- c(
  "age",
  "BMI",
  "pulse",
  "BUN",
  "TC",
  "UA",
  "HbA1c"
)

stopifnot(
  identical(
    final_selected_predictors,
    final_predictors
  )
)
# ==============================================================================
# 12. Save final feature-selection summary
# ==============================================================================

feature_selection_summary <- data.frame(
  Stage = c(
    "Initial candidate variables",
    "Diagnostic variables excluded before LASSO",
    "Variables entering LASSO",
    "Nonzero predictors at lambda.1se",
    "Predictors after collinearity assessment"
  ),
  N = c(
    20,
    6,
    14,
    8,
    7
  )
)

write.csv(
  feature_selection_summary,
  file.path(
    results_dir,
    "feature_selection_summary.csv"
  ),
  row.names = FALSE
)


final_predictor_table <- data.frame(
  Predictor = final_selected_predictors
)

write.csv(
  final_predictor_table,
  file.path(
    results_dir,
    "final_predictors.csv"
  ),
  row.names = FALSE
)
# ==============================================================================
# 13. Reproducibility summary
# ==============================================================================

cat("\n========================================\n")
cat("Feature-selection reproducibility check\n")
cat("========================================\n")

cat(
  "lambda.1se =",
  format(lambda_1se, digits = 10),
  "\n"
)

cat(
  "Number of nonzero predictors =",
  nrow(lasso_selected),
  "\n"
)

cat(
  "Selected predictors:",
  paste(
    lasso_selected$Predictor,
    collapse = ", "
  ),
  "\n"
)

cat(
  "Correlation between TC and LDL-C =",
  round(tc_ldl_correlation, 4),
  "\n"
)

cat(
  "Final predictors:",
  paste(
    final_selected_predictors,
    collapse = ", "
  ),
  "\n"
)

cat("========================================\n")