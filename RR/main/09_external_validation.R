# ==============================================================================
# 09_external_validation.R
#
# External validation of the locked CatBoost model
#
# IMPORTANT:
# - No model retraining is performed.
# - No threshold optimization is performed.
# - The predefined threshold from the development cohort is applied.
# ==============================================================================


library(dplyr)
library(pROC)
library(PRROC)
library(catboost)

source("RR/main/00_setup.R")
# ==============================================================================
# 1. Load locked model and external-validation dataset
# ==============================================================================


cat_model_path <- file.path(
  model_dir,
  "locked",
  "model_cat.rds"
)


cat_model <- readRDS(
  cat_model_path
)


external_data <- readRDS(
  file.path(
    processed_dir,
    "external_validation_model_data.rds"
  )
)


cat(
  "External validation dataset:",
  nrow(external_data),
  "participants\n"
)
# ==============================================================================
# 2. Define final predictors
# ==============================================================================


final_predictors <- c(
  "age",
  "BMI",
  "pulse",
  "BUN",
  "TC",
  "UA",
  "HbA1c"
)


stopifnot(
  all(
    final_predictors %in%
      names(external_data)
  )
)


X_external <- external_data %>%
  dplyr::select(
    all_of(final_predictors)
  )


y_external <- external_data$met_diagnosis
# ==============================================================================
# 3. Generate predicted probabilities
# ==============================================================================


# ==============================================================================
# 3. Generate predicted probabilities
# ==============================================================================


external_pool <- catboost.load_pool(
  data = X_external
)


external_probability <- catboost.predict(
  cat_model,
  external_pool,
  prediction_type = "Probability"
)
head(external_probability)


locked_threshold <- 0.2498348


external_prediction <- ifelse(
  external_probability >= locked_threshold,
  1,
  0
)
# ==============================================================================
# 4. Discrimination
# ==============================================================================


roc_external <- pROC::roc(
  response = y_external,
  predictor = external_probability,
  quiet = TRUE
)


auc_external <- as.numeric(
  pROC::auc(roc_external)
)


auc_ci_external <- pROC::ci.auc(
  roc_external
)


cat("\n")
cat("External-validation discrimination\n")
cat("-----------------------------------\n")
cat(
  "AUC =",
  round(auc_external, 3),
  "\n"
)


cat(
  "95% CI =",
  paste(
    round(auc_ci_external[1], 3),
    round(auc_ci_external[3], 3),
    sep = " - "
  ),
  "\n"
)
# ==============================================================================
# 5. Precision-recall performance and classification metrics
# ==============================================================================


pr_external <- PRROC::pr.curve(
  scores.class0 = external_probability[y_external == 1],
  scores.class1 = external_probability[y_external == 0],
  curve = FALSE
)


pr_auc_external <- pr_external$auc.integral


TP <- sum(
  external_prediction == 1 &
    y_external == 1
)

TN <- sum(
  external_prediction == 0 &
    y_external == 0
)

FP <- sum(
  external_prediction == 1 &
    y_external == 0
)

FN <- sum(
  external_prediction == 0 &
    y_external == 1
)


sensitivity_external <- TP / (TP + FN)

specificity_external <- TN / (TN + FP)

accuracy_external <- (TP + TN) /
  length(y_external)

ppv_external <- TP / (TP + FP)

npv_external <- TN / (TN + FN)

f1_external <- 2 * TP /
  (2 * TP + FP + FN)


cat("\n")
cat("External-validation classification performance\n")
cat("----------------------------------------------\n")

cat(
  "PR-AUC =",
  round(pr_auc_external, 3),
  "\n"
)

cat(
  "Threshold =",
  locked_threshold,
  "\n"
)

cat(
  "Sensitivity =",
  round(sensitivity_external, 3),
  "\n"
)

cat(
  "Specificity =",
  round(specificity_external, 3),
  "\n"
)

cat(
  "Accuracy =",
  round(accuracy_external, 3),
  "\n"
)

cat(
  "PPV =",
  round(ppv_external, 3),
  "\n"
)

cat(
  "NPV =",
  round(npv_external, 3),
  "\n"
)

cat(
  "F1 =",
  round(f1_external, 3),
  "\n"
)
# ==============================================================================
# 6. Calibration
# ==============================================================================


# Brier score
brier_external <- mean(
  (external_probability - y_external)^2
)


# Avoid infinite logits in the unlikely event of probabilities equal to 0 or 1
eps <- 1e-15

external_probability_safe <- pmin(
  pmax(external_probability, eps),
  1 - eps
)


external_logit <- qlogis(
  external_probability_safe
)


# Calibration-in-the-large (CITL)
calibration_intercept_model <- glm(
  y_external ~ 1,
  family = binomial(),
  offset = external_logit
)

citl_external <- as.numeric(
  coef(calibration_intercept_model)[1]
)


# Calibration intercept and slope
calibration_model <- glm(
  y_external ~ external_logit,
  family = binomial()
)

calibration_intercept_external <- as.numeric(
  coef(calibration_model)[1]
)

calibration_slope_external <- as.numeric(
  coef(calibration_model)[2]
)


# Observed-to-expected ratio
observed_events <- sum(y_external)

expected_events <- sum(external_probability)

oe_ratio_external <- observed_events /
  expected_events


cat("\n")
cat("External-validation calibration\n")
cat("--------------------------------\n")

cat(
  "Brier =",
  round(brier_external, 3),
  "\n"
)

cat(
  "Calibration intercept =",
  round(calibration_intercept_external, 3),
  "\n"
)

cat(
  "Calibration slope =",
  round(calibration_slope_external, 3),
  "\n"
)

cat(
  "CITL =",
  round(citl_external, 3),
  "\n"
)

cat(
  "O:E ratio =",
  round(oe_ratio_external, 3),
  "\n"
)
# ==============================================================================
# 7. Save aggregate external-validation results
# ==============================================================================


external_validation_summary <- data.frame(
  N = length(y_external),
  MetS_cases = sum(y_external == 1),
  Non_MetS = sum(y_external == 0),
  Prevalence = mean(y_external == 1),
  AUC = auc_external,
  AUC_CI_lower = as.numeric(auc_ci_external[1]),
  AUC_CI_upper = as.numeric(auc_ci_external[3]),
  PR_AUC = pr_auc_external,
  Threshold = locked_threshold,
  Sensitivity = sensitivity_external,
  Specificity = specificity_external,
  Accuracy = accuracy_external,
  PPV = ppv_external,
  NPV = npv_external,
  F1 = f1_external,
  Brier = brier_external,
  Calibration_intercept = calibration_intercept_external,
  Calibration_slope = calibration_slope_external,
  CITL = citl_external,
  OE_ratio = oe_ratio_external
)


write.csv(
  external_validation_summary,
  file.path(
    results_dir,
    "external_validation_summary.csv"
  ),
  row.names = FALSE
)
# ==============================================================================
# 8. Final reproducibility summary
# ==============================================================================


cat("\n")
cat("============================================================\n")
cat("External-validation reproducibility summary\n")
cat("============================================================\n")

cat("N =", length(y_external), "\n")
cat("MetS cases =", sum(y_external == 1), "\n")
cat(
  "MetS prevalence =",
  round(100 * mean(y_external == 1), 1),
  "%\n"
)

cat(
  "AUC =",
  round(auc_external, 3),
  "(",
  round(auc_ci_external[1], 3),
  "-",
  round(auc_ci_external[3], 3),
  ")\n"
)

cat("PR-AUC =", round(pr_auc_external, 3), "\n")
cat("Threshold =", locked_threshold, "\n")
cat("Sensitivity =", round(sensitivity_external, 3), "\n")
cat("Specificity =", round(specificity_external, 3), "\n")
cat("Accuracy =", round(accuracy_external, 3), "\n")
cat("PPV =", round(ppv_external, 3), "\n")
cat("NPV =", round(npv_external, 3), "\n")
cat("F1 =", round(f1_external, 3), "\n")
cat("Brier =", round(brier_external, 3), "\n")
cat(
  "Calibration intercept =",
  round(calibration_intercept_external, 3),
  "\n"
)
cat(
  "Calibration slope =",
  round(calibration_slope_external, 3),
  "\n"
)
cat("CITL =", round(citl_external, 3), "\n")
cat("O:E ratio =", round(oe_ratio_external, 3), "\n")

cat("Outcome-based sampling performed: No\n")
cat("Model retraining performed: No\n")
cat("Threshold re-optimization performed: No\n")

cat("============================================================\n")