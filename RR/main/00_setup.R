# ==============================================================================
# 00_setup.R
# Common settings for the MetS machine-learning analysis
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. Reproducibility
# ------------------------------------------------------------------------------

SEED <- 123
set.seed(SEED)


# ------------------------------------------------------------------------------
# 2. Project directories
# ------------------------------------------------------------------------------

# The analysis should be run from the repository root ("代码整理").
# Individual-level raw data are not included in the public repository.

charls_raw_dir <- file.path("data", "raw", "CHARLS2015")
processed_dir  <- file.path("data", "processed")
results_dir    <- "results"
model_dir      <- "model"

dir.create(processed_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(results_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(model_dir, showWarnings = FALSE, recursive = TRUE)


# ------------------------------------------------------------------------------
# 3. Final model predictors
# ------------------------------------------------------------------------------

final_predictors <- c(
  "age",
  "BMI",
  "pulse",
  "BUN",
  "TC",
  "UA",
  "HbA1c"
)


# ------------------------------------------------------------------------------
# 4. MetS diagnostic variables excluded from predictor selection
# ------------------------------------------------------------------------------

mets_diagnostic_variables <- c(
  "waist",
  "SBP",
  "DBP",
  "HDL_C",
  "TG",
  "FPG"
)


# ------------------------------------------------------------------------------
# 5. Candidate predictors entering LASSO
# ------------------------------------------------------------------------------

lasso_candidate_variables <- c(
  "age",
  "gender",
  "smoking",
  "drinking",
  "BMI",
  "pulse",
  "grip",
  "CRP",
  "BUN",
  "LDL_C",
  "TC",
  "UA",
  "CREA",
  "HbA1c"
)
external_raw_dir <- file.path("data", "raw", "external_validation")