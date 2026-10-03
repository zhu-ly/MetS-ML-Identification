# ==============================================================================
# 01_verify_native_catboost_model.R
#
# Export the locked final CatBoost model from the RDS object to CatBoost's
# native binary model format (.cbm), reload it, and verify prediction
# equivalence using the synthetic example inputs.
#
# IMPORTANT:
# - The model is NOT retrained.
# - The locked RDS model is NOT modified.
# - No study participant-level data are used.
# ==============================================================================


library(catboost)


# ==============================================================================
# 1. File paths
# ==============================================================================

rds_model_file <- file.path(
  "model",
  "locked",
  "model_cat.rds"
)

cbm_model_file <- file.path(
  "model",
  "locked",
  "model_cat.cbm"
)

example_file <- file.path(
  "example",
  "example_input.csv"
)


if (!file.exists(rds_model_file)) {
  stop(
    paste0(
      "Cannot find locked RDS model: ",
      rds_model_file
    )
  )
}

if (!file.exists(example_file)) {
  stop(
    paste0(
      "Cannot find synthetic example input: ",
      example_file
    )
  )
}


# ==============================================================================
# 2. Required predictors
# ==============================================================================

required_predictors <- c(
  "age",
  "BMI",
  "pulse",
  "BUN",
  "TC",
  "UA",
  "HbA1c"
)


# ==============================================================================
# 3. Read and validate synthetic example input
# ==============================================================================

example_data <- read.csv(
  example_file,
  check.names = FALSE
)


if (
  !identical(
    names(example_data),
    required_predictors
  )
) {
  
  stop(
    paste0(
      "Predictor names or order are incorrect.\n",
      "Required order: ",
      paste(
        required_predictors,
        collapse = ", "
      )
    )
  )
}


if (
  anyNA(
    example_data[
      ,
      required_predictors,
      drop = FALSE
    ]
  )
) {
  
  stop(
    "Synthetic example input contains missing values."
  )
}


prediction_matrix <- as.matrix(
  example_data[
    ,
    required_predictors,
    drop = FALSE
  ]
)


# CatBoost R expects numerical matrix data in double precision.

storage.mode(
  prediction_matrix
) <- "double"


if (!is.double(prediction_matrix)) {
  stop(
    "Prediction matrix was not converted to double precision."
  )
}


prediction_pool <- catboost.load_pool(
  data = prediction_matrix
)


# ==============================================================================
# 4. Load original locked RDS model
# ==============================================================================

cat(
  "Loading original locked RDS model...\n"
)


model_rds <- readRDS(
  rds_model_file
)


# ==============================================================================
# 5. Generate reference predictions from locked RDS model
# ==============================================================================

prob_rds <- as.numeric(
  catboost.predict(
    model_rds,
    prediction_pool,
    prediction_type = "Probability"
  )
)


# Previously established prediction fingerprint.
# These values were generated directly from the locked manuscript model.

expected_fingerprint <- c(
  0.0225694136,
  0.1514732287,
  0.7957659531
)


fingerprint_difference <- abs(
  prob_rds -
    expected_fingerprint
)


if (
  any(
    fingerprint_difference >
    1e-9
  )
) {
  
  stop(
    paste0(
      "Locked RDS model no longer reproduces the established ",
      "prediction fingerprint."
    )
  )
}


cat(
  "Locked RDS prediction fingerprint verified.\n"
)


# ==============================================================================
# 6. Export the SAME locked model to CatBoost native binary format
#
# This does NOT retrain or refit the model.
# ==============================================================================

cat(
  "Exporting locked model to native CatBoost format...\n"
)


catboost.save_model(
  model_rds,
  cbm_model_file
)


if (!file.exists(cbm_model_file)) {
  stop(
    "Native CatBoost model file was not created."
  )
}


cbm_file_size <- file.info(
  cbm_model_file
)$size


if (
  is.na(cbm_file_size) ||
  cbm_file_size <= 0
) {
  
  stop(
    "Native CatBoost model file is empty or invalid."
  )
}


# ==============================================================================
# 7. Reload native .cbm model
# ==============================================================================

cat(
  "Reloading native CatBoost model...\n"
)


model_cbm <- catboost.load_model(
  cbm_model_file
)


# ==============================================================================
# 8. Predict using reloaded native model
# ==============================================================================

prob_cbm <- as.numeric(
  catboost.predict(
    model_cbm,
    prediction_pool,
    prediction_type = "Probability"
  )
)


if (
  length(prob_cbm) !=
  length(prob_rds)
) {
  
  stop(
    "RDS and CBM models generated different numbers of predictions."
  )
}


# ==============================================================================
# 9. Compare predictions
# ==============================================================================

absolute_difference <- abs(
  prob_rds -
    prob_cbm
)


max_absolute_difference <- max(
  absolute_difference
)


comparison <- data.frame(
  
  Example_ID =
    seq_along(prob_rds),
  
  RDS_Probability =
    prob_rds,
  
  CBM_Probability =
    prob_cbm,
  
  Absolute_Difference =
    absolute_difference
)


# Use a strict numerical tolerance.

tolerance <- 1e-12


equivalent <- all(
  absolute_difference <=
    tolerance
)


if (!equivalent) {
  
  stop(
    paste0(
      "Prediction equivalence check failed. ",
      "Maximum absolute difference = ",
      format(
        max_absolute_difference,
        scientific = TRUE
      )
    )
  )
}


# ==============================================================================
# 10. Verify native model against established fingerprint
# ==============================================================================

cbm_fingerprint_difference <- abs(
  prob_cbm -
    expected_fingerprint
)


if (
  any(
    cbm_fingerprint_difference >
    1e-9
  )
) {
  
  stop(
    "Reloaded CBM model does not reproduce the established fingerprint."
  )
}


# ==============================================================================
# 11. Console summary
# ==============================================================================

cat("\n")
cat("============================================================\n")
cat("Native CatBoost model equivalence verification\n")
cat("============================================================\n")


print(
  comparison,
  row.names = FALSE,
  digits = 12
)


cat("\n")
cat("Prediction fingerprint\n")
cat("----------------------\n")


for (
  i in seq_along(prob_rds)
) {
  
  cat(
    paste0(
      "Example ",
      i,
      "\n",
      "  RDS: ",
      sprintf(
        "%.10f",
        prob_rds[i]
      ),
      "\n",
      "  CBM: ",
      sprintf(
        "%.10f",
        prob_cbm[i]
      ),
      "\n"
    )
  )
}


cat("\n")
cat("Equivalence checks\n")
cat("------------------\n")


cat(
  "Original locked RDS model used: Yes\n"
)

cat(
  "Model retrained: No\n"
)

cat(
  "Native CBM model exported: Yes\n"
)

cat(
  "Native CBM model reloaded: Yes\n"
)

cat(
  "Synthetic example input used: Yes\n"
)

cat(
  "RDS fingerprint reproduced: Yes\n"
)

cat(
  "CBM fingerprint reproduced: Yes\n"
)

cat(
  "Maximum absolute RDS-CBM difference =",
  format(
    max_absolute_difference,
    scientific = TRUE,
    digits = 6
  ),
  "\n"
)

cat(
  "Tolerance =",
  format(
    tolerance,
    scientific = TRUE
  ),
  "\n"
)

cat(
  "RDS-CBM predictions equivalent: Yes\n"
)

cat(
  "Participant-level study data used: No\n"
)

cat(
  "Native model file:",
  cbm_model_file,
  "\n"
)

cat(
  "Native model file size:",
  cbm_file_size,
  "bytes\n"
)

cat("============================================================\n")