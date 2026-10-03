# ==============================================================================
# 00_create_example_input.R
#
# Create synthetic example inputs for demonstrating prediction with the
# locked final CatBoost model.
#
# IMPORTANT:
# - These are synthetic examples.
# - They are not real CHARLS participants.
# - They are not real external-validation participants.
# - They must not be interpreted as clinical recommendations.
# ==============================================================================

source("RR/main/00_setup.R")

example_input <- data.frame(
  
  age = c(
    50,
    60,
    70
  ),
  
  BMI = c(
    21,
    24,
    28
  ),
  
  pulse = c(
    65,
    75,
    85
  ),
  
  BUN = c(
    12,
    16,
    20
  ),
  
  TC = c(
    160,
    190,
    220
  ),
  
  UA = c(
    4,
    5,
    6
  ),
  
  HbA1c = c(
    5.0,
    6.0,
    7.0
  )
)


# Predictor names and order must exactly match the final model specification.

stopifnot(
  identical(
    names(example_input),
    final_predictors
  )
)


dir.create(
  "example",
  showWarnings = FALSE,
  recursive = TRUE
)


write.csv(
  example_input,
  "example/example_input.csv",
  row.names = FALSE
)


cat("\n")
cat("============================================================\n")
cat("Synthetic example input created\n")
cat("============================================================\n")

print(example_input)

cat("\n")
cat(
  "Rows =",
  nrow(example_input),
  "\n"
)

cat(
  "Predictors =",
  paste(
    names(example_input),
    collapse = ", "
  ),
  "\n"
)

cat(
  "Real participant data used: No\n"
)

cat(
  "Output: example/example_input.csv\n"
)

cat("============================================================\n")