# ==============================================================================
# predict_example.R
#
# Minimal prediction example using the locked final CatBoost model.
#
# This script:
# 1. Reads synthetic example inputs.
# 2. Loads the locked final 7-predictor CatBoost model.
# 3. Checks predictor names, order, and values.
# 4. Converts predictor data explicitly to double precision.
# 5. Generates predicted probabilities of prevalent metabolic syndrome (MetS).
#
# IMPORTANT:
# - The example data are synthetic.
# - No real participant-level data are required.
# - The model identifies/classifies prevalent MetS and does not predict
#   future MetS incidence.
# ==============================================================================


library(catboost)


# ==============================================================================
# 1. File paths
# ==============================================================================

input_file <- file.path(
  "example",
  "example_input.csv"
)

model_file <- file.path(
  "model",
  "locked",
  "model_cat.rds"
)


if (!file.exists(input_file)) {
  stop(
    paste0(
      "Cannot find example input file: ",
      input_file
    )
  )
}


if (!file.exists(model_file)) {
  stop(
    paste0(
      "Cannot find locked CatBoost model: ",
      model_file
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
# 3. Read synthetic example data
# ==============================================================================

new_data <- read.csv(
  input_file,
  check.names = FALSE
)


if (nrow(new_data) == 0) {
  stop(
    "The example input file contains no observations."
  )
}


if (
  !identical(
    names(new_data),
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


# ==============================================================================
# 4. Validate predictor values
# ==============================================================================

non_numeric <- required_predictors[
  !vapply(
    new_data[
      ,
      required_predictors,
      drop = FALSE
    ],
    is.numeric,
    logical(1)
  )
]


if (length(non_numeric) > 0) {
  
  stop(
    paste0(
      "The following predictors are not numeric: ",
      paste(
        non_numeric,
        collapse = ", "
      )
    )
  )
}


if (
  anyNA(
    new_data[
      ,
      required_predictors,
      drop = FALSE
    ]
  )
) {
  
  stop(
    "Missing predictor values are not allowed in this prediction example."
  )
}


if (
  !all(
    vapply(
      new_data[
        ,
        required_predictors,
        drop = FALSE
      ],
      function(x) {
        all(
          is.finite(x)
        )
      },
      logical(1)
    )
  )
) {
  
  stop(
    "All predictor values must be finite numeric values."
  )
}


# ==============================================================================
# 5. Construct CatBoost input matrix
#
# IMPORTANT:
# read.csv() may import whole-number columns as integer.
# The R CatBoost interface expects a double/numeric matrix here.
# Therefore, storage mode is explicitly converted to double.
# ==============================================================================

prediction_matrix <- as.matrix(
  new_data[
    ,
    required_predictors,
    drop = FALSE
  ]
)


storage.mode(
  prediction_matrix
) <- "double"


if (
  !is.double(
    prediction_matrix
  )
) {
  
  stop(
    "Failed to convert prediction matrix to double precision."
  )
}


# ==============================================================================
# 6. Load locked final CatBoost model
# ==============================================================================

model_cat <- readRDS(
  model_file
)


# ==============================================================================
# 7. Generate predicted probabilities
# ==============================================================================

prediction_pool <- catboost.load_pool(
  data = prediction_matrix
)


predicted_probability <- as.numeric(
  catboost.predict(
    model_cat,
    prediction_pool,
    prediction_type = "Probability"
  )
)


if (
  length(predicted_probability) !=
  nrow(new_data)
) {
  
  stop(
    "Number of predictions does not match number of input observations."
  )
}


if (
  any(
    !is.finite(
      predicted_probability
    )
  ) ||
  any(
    predicted_probability < 0 |
    predicted_probability > 1
  )
) {
  
  stop(
    "Invalid predicted probability generated."
  )
}


# ==============================================================================
# 8. Create prediction output
# ==============================================================================

prediction_output <- data.frame(
  
  Example_ID =
    seq_len(
      nrow(new_data)
    ),
  
  new_data,
  
  Predicted_Probability =
    predicted_probability,
  
  check.names = FALSE
)


# ==============================================================================
# 9. Display prediction results
# ==============================================================================

cat("\n")
cat("============================================================\n")
cat("Locked CatBoost prediction example\n")
cat("============================================================\n")


print(
  prediction_output,
  row.names = FALSE,
  digits = 10
)


cat("\n")
cat("Prediction fingerprint\n")
cat("----------------------\n")


for (
  i in seq_along(
    predicted_probability
  )
) {
  
  cat(
    paste0(
      "Example ",
      i,
      ": ",
      sprintf(
        "%.10f",
        predicted_probability[i]
      ),
      "\n"
    )
  )
}


cat("\n")
cat("Reproducibility checks\n")
cat("----------------------\n")


cat(
  "Locked CatBoost model loaded: Yes\n"
)

cat(
  "Required predictor order verified: Yes\n"
)

cat(
  "Prediction matrix converted to double: Yes\n"
)

cat(
  "Synthetic example input used: Yes\n"
)

cat(
  "Number of input rows =",
  nrow(new_data),
  "\n"
)

cat(
  "Number of predictions =",
  length(
    predicted_probability
  ),
  "\n"
)

cat(
  "Prediction probabilities valid: Yes\n"
)

cat(
  "Participant-level study data required: No\n"
)

cat("============================================================\n")