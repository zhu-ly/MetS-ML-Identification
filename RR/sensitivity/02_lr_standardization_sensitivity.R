# ==============================================================================
# 02_lr_standardization_sensitivity.R
#
# Sensitivity analysis:
# Unstandardized versus Z-score standardized logistic regression
#
# Objectives:
# 1. Obtain predictions from the locked original unstandardized LR model.
# 2. Z-score standardize the seven predictors using means and SDs derived
#    exclusively from the training set.
# 3. Apply the same training-derived scaling parameters to the testing set.
# 4. Refit logistic regression using standardized predictors.
# 5. Compare testing-set performance and predicted probabilities.
#
# IMPORTANT:
# - Scaling parameters are derived from the training set only.
# - The testing set is never used to determine scaling parameters.
# - Classification thresholds are determined in the training set using
#   Youden's index.
# ==============================================================================


library(dplyr)
library(pROC)


source("RR/main/00_setup.R")


set.seed(SEED)


# ==============================================================================
# 1. Load development datasets and locked original LR model
# ==============================================================================


train_set <- readRDS(
  file.path(
    processed_dir,
    "train_set.rds"
  )
)


test_set <- readRDS(
  file.path(
    processed_dir,
    "test_set.rds"
  )
)


model_lr <- readRDS(
  file.path(
    model_dir,
    "locked",
    "model_lr.rds"
  )
)


final_vars <- c(
  "age",
  "BMI",
  "pulse",
  "BUN",
  "TC",
  "UA",
  "HbA1c"
)


required_vars <- c(
  final_vars,
  "met_diagnosis"
)


missing_train <- setdiff(
  required_vars,
  names(train_set)
)


missing_test <- setdiff(
  required_vars,
  names(test_set)
)


if (length(missing_train) > 0) {
  stop(
    paste(
      "Training dataset is missing:",
      paste(missing_train, collapse = ", ")
    )
  )
}


if (length(missing_test) > 0) {
  stop(
    paste(
      "Testing dataset is missing:",
      paste(missing_test, collapse = ", ")
    )
  )
}


stopifnot(
  nrow(train_set) == 6907,
  nrow(test_set) == 2960
)


# ==============================================================================
# 2. Convert outcome to numeric 0/1
# ==============================================================================


to_binary <- function(x) {
  
  x_char <- as.character(x)
  
  unique_values <- unique(
    na.omit(x_char)
  )
  
  
  if (
    all(
      unique_values %in% c("0", "1")
    )
  ) {
    
    return(
      as.integer(x_char)
    )
  }
  
  
  if (
    all(
      unique_values %in% c(
        "Control",
        "Case"
      )
    )
  ) {
    
    return(
      ifelse(
        x_char == "Case",
        1L,
        0L
      )
    )
  }
  
  
  stop(
    "Unrecognized met_diagnosis coding."
  )
}


train_y <- to_binary(
  train_set$met_diagnosis
)


test_y <- to_binary(
  test_set$met_diagnosis
)


stopifnot(
  all(train_y %in% c(0, 1)),
  all(test_y %in% c(0, 1)),
  sum(train_y) == 1538,
  sum(test_y) == 659
)


# ==============================================================================
# 3. Obtain original unstandardized LR probabilities
#
# The locked LR model is a caret::train object.
# Therefore, probabilities are obtained using type = "prob".
# ==============================================================================


original_lr_train_prob_raw <- predict(
  model_lr,
  newdata = train_set,
  type = "prob"
)


original_lr_test_prob_raw <- predict(
  model_lr,
  newdata = test_set,
  type = "prob"
)


# Identify the probability column corresponding to MetS = 1.
# In the locked caret LR model, the outcome levels are Control and Case,
# with Case representing MetS = 1.

if (
  "Case" %in% names(original_lr_train_prob_raw)
) {
  
  positive_probability_column <- "Case"
  
} else if (
  "1" %in% names(original_lr_train_prob_raw)
) {
  
  positive_probability_column <- "1"
  
} else if (
  "X1" %in% names(original_lr_train_prob_raw)
) {
  
  positive_probability_column <- "X1"
  
} else {
  
  stop(
    paste0(
      "Cannot identify the positive-class probability column. ",
      "Available columns: ",
      paste(
        names(original_lr_train_prob_raw),
        collapse = ", "
      )
    )
  )
}


original_lr_train_prob <- as.numeric(
  original_lr_train_prob_raw[[positive_probability_column]]
)


original_lr_test_prob <- as.numeric(
  original_lr_test_prob_raw[[positive_probability_column]]
)


stopifnot(
  length(original_lr_train_prob) == nrow(train_set),
  length(original_lr_test_prob) == nrow(test_set),
  all(
    original_lr_train_prob >= 0 &
      original_lr_train_prob <= 1
  ),
  all(
    original_lr_test_prob >= 0 &
      original_lr_test_prob <= 1
  )
)


# ==============================================================================
# 4. Determine original LR training-set Youden threshold
# ==============================================================================


roc_original_train <- pROC::roc(
  response = train_y,
  predictor = original_lr_train_prob,
  levels = c(0, 1),
  direction = "<",
  quiet = TRUE
)


original_lr_threshold <- as.numeric(
  pROC::coords(
    roc_original_train,
    x = "best",
    best.method = "youden",
    ret = "threshold",
    transpose = FALSE
  )$threshold[1]
)


# ==============================================================================
# 5. Calculate training-derived Z-score parameters
# ==============================================================================


train_means <- sapply(
  train_set[
    ,
    final_vars
  ],
  mean,
  na.rm = TRUE
)


train_sds <- sapply(
  train_set[
    ,
    final_vars
  ],
  sd,
  na.rm = TRUE
)


if (
  any(
    !is.finite(train_means)
  )
) {
  
  stop(
    "Non-finite training-set means detected."
  )
}


if (
  any(
    !is.finite(train_sds)
  ) ||
  any(
    train_sds == 0
  )
) {
  
  stop(
    "Invalid training-set standard deviations detected."
  )
}


# ==============================================================================
# 6. Standardize training and testing predictors
#
# The testing set uses the means and SDs calculated from the TRAINING set.
# ==============================================================================


train_std <- as.data.frame(
  scale(
    train_set[
      ,
      final_vars
    ],
    center = train_means,
    scale = train_sds
  )
)


test_std <- as.data.frame(
  scale(
    test_set[
      ,
      final_vars
    ],
    center = train_means,
    scale = train_sds
  )
)


train_std$MetS <- train_y
test_std$MetS <- test_y


stopifnot(
  !anyNA(train_std),
  !anyNA(test_std)
)


# ==============================================================================
# 7. Fit standardized logistic regression
# ==============================================================================


standardized_lr_model <- glm(
  MetS ~ .,
  data = train_std,
  family = binomial()
)


standardized_lr_train_prob <- as.numeric(
  predict(
    standardized_lr_model,
    newdata = train_std,
    type = "response"
  )
)


standardized_lr_test_prob <- as.numeric(
  predict(
    standardized_lr_model,
    newdata = test_std,
    type = "response"
  )
)


# ==============================================================================
# 8. Determine standardized LR training-set Youden threshold
# ==============================================================================


roc_standardized_train <- pROC::roc(
  response = train_y,
  predictor = standardized_lr_train_prob,
  levels = c(0, 1),
  direction = "<",
  quiet = TRUE
)


standardized_lr_threshold <- as.numeric(
  pROC::coords(
    roc_standardized_train,
    x = "best",
    best.method = "youden",
    ret = "threshold",
    transpose = FALSE
  )$threshold[1]
)


# ==============================================================================
# 9. Performance-evaluation function
# ==============================================================================


evaluate_lr <- function(
    probability,
    outcome,
    threshold,
    model_name
) {
  
  roc_object <- pROC::roc(
    response = outcome,
    predictor = probability,
    levels = c(0, 1),
    direction = "<",
    quiet = TRUE
  )
  
  
  auc_value <- as.numeric(
    pROC::auc(
      roc_object
    )
  )
  
  
  auc_ci <- as.numeric(
    pROC::ci.auc(
      roc_object,
      method = "delong"
    )
  )
  
  
  predicted_class <- ifelse(
    probability >= threshold,
    1L,
    0L
  )
  
  
  TP <- sum(
    predicted_class == 1 &
      outcome == 1
  )
  
  
  TN <- sum(
    predicted_class == 0 &
      outcome == 0
  )
  
  
  FP <- sum(
    predicted_class == 1 &
      outcome == 0
  )
  
  
  FN <- sum(
    predicted_class == 0 &
      outcome == 1
  )
  
  
  accuracy <- (
    TP + TN
  ) / length(outcome)
  
  
  sensitivity <- TP / (
    TP + FN
  )
  
  
  specificity <- TN / (
    TN + FP
  )
  
  
  ppv <- TP / (
    TP + FP
  )
  
  
  npv <- TN / (
    TN + FN
  )
  
  
  f1 <- 2 * TP / (
    2 * TP +
      FP +
      FN
  )
  
  
  brier <- mean(
    (
      probability -
        outcome
    )^2
  )
  
  
  data.frame(
    Model = model_name,
    Threshold = threshold,
    AUC = auc_value,
    AUC_CI_Lower = auc_ci[1],
    AUC_CI_Upper = auc_ci[3],
    Brier = brier,
    Accuracy = accuracy,
    Sensitivity = sensitivity,
    Specificity = specificity,
    PPV = ppv,
    NPV = npv,
    F1 = f1,
    stringsAsFactors = FALSE
  )
}


# ==============================================================================
# 10. Evaluate both LR models in the independent testing set
# ==============================================================================


original_lr_result <- evaluate_lr(
  probability = original_lr_test_prob,
  outcome = test_y,
  threshold = original_lr_threshold,
  model_name = "Original LR"
)


standardized_lr_result <- evaluate_lr(
  probability = standardized_lr_test_prob,
  outcome = test_y,
  threshold = standardized_lr_threshold,
  model_name = "Standardized LR"
)


# ==============================================================================
# 11. Compare predicted probabilities
# ==============================================================================


absolute_probability_difference <- abs(
  original_lr_test_prob -
    standardized_lr_test_prob
)


max_abs_probability_difference <- max(
  absolute_probability_difference
)


mean_abs_probability_difference <- mean(
  absolute_probability_difference
)


# ==============================================================================
# 12. Combine aggregate results
# ==============================================================================


lr_sensitivity_table <- dplyr::bind_rows(
  original_lr_result,
  standardized_lr_result
)


lr_sensitivity_table$Max_Absolute_Probability_Difference <- c(
  NA_real_,
  max_abs_probability_difference
)


lr_sensitivity_table$Mean_Absolute_Probability_Difference <- c(
  NA_real_,
  mean_abs_probability_difference
)


# ==============================================================================
# 13. Save aggregate reproducibility outputs
# ==============================================================================


write.csv(
  lr_sensitivity_table,
  file.path(
    results_dir,
    "LR_Standardization_Sensitivity.csv"
  ),
  row.names = FALSE
)


scaling_parameters <- data.frame(
  Predictor = final_vars,
  Training_Mean = as.numeric(
    train_means[
      final_vars
    ]
  ),
  Training_SD = as.numeric(
    train_sds[
      final_vars
    ]
  ),
  stringsAsFactors = FALSE
)


write.csv(
  scaling_parameters,
  file.path(
    results_dir,
    "LR_Standardization_Training_Scaling_Parameters.csv"
  ),
  row.names = FALSE
)


# ==============================================================================
# 14. Reproducibility checks
# ==============================================================================


stopifnot(
  abs(
    original_lr_result$AUC -
      0.822
  ) < 0.001,
  
  abs(
    original_lr_result$Brier -
      0.133
  ) < 0.001,
  
  abs(
    standardized_lr_result$AUC -
      0.822
  ) < 0.001,
  
  abs(
    standardized_lr_result$Brier -
      0.133
  ) < 0.001
)


# Standardization of the same linear predictors should produce
# numerically equivalent fitted probabilities, up to floating-point error.

if (
  max_abs_probability_difference >
  1e-10
) {
  
  stop(
    paste0(
      "Standardized and unstandardized LR probabilities differ ",
      "more than expected. Maximum absolute difference = ",
      format(
        max_abs_probability_difference,
        scientific = TRUE
      )
    )
  )
}


# ==============================================================================
# 15. Final console summary
# ==============================================================================


cat("\n")
cat("============================================================\n")
cat("Logistic-regression standardization sensitivity analysis\n")
cat("============================================================\n")


cat(
  "Training N =",
  nrow(train_set),
  "\n"
)


cat(
  "Training MetS cases =",
  sum(train_y),
  "\n"
)


cat(
  "Testing N =",
  nrow(test_set),
  "\n"
)


cat(
  "Testing MetS cases =",
  sum(test_y),
  "\n"
)


cat("\n")
cat("Testing-set performance\n")
cat("-----------------------\n")


console_table <- lr_sensitivity_table %>%
  mutate(
    AUC_95CI = sprintf(
      "%.3f (%.3f-%.3f)",
      AUC,
      AUC_CI_Lower,
      AUC_CI_Upper
    ),
    Brier = round(
      Brier,
      3
    ),
    Accuracy = round(
      Accuracy,
      3
    ),
    Sensitivity = round(
      Sensitivity,
      3
    ),
    Specificity = round(
      Specificity,
      3
    ),
    F1 = round(
      F1,
      3
    )
  ) %>%
  select(
    Model,
    AUC_95CI,
    Brier,
    Accuracy,
    Sensitivity,
    Specificity,
    F1
  )


print(
  console_table
)


cat("\n")
cat("Training-set Youden thresholds\n")
cat("------------------------------\n")


cat(
  "Original LR threshold =",
  sprintf(
    "%.7f",
    original_lr_threshold
  ),
  "\n"
)


cat(
  "Standardized LR threshold =",
  sprintf(
    "%.7f",
    standardized_lr_threshold
  ),
  "\n"
)


cat("\n")
cat("Prediction equivalence\n")
cat("----------------------\n")


cat(
  "Maximum absolute probability difference =",
  format(
    max_abs_probability_difference,
    scientific = TRUE,
    digits = 6
  ),
  "\n"
)


cat(
  "Mean absolute probability difference =",
  format(
    mean_abs_probability_difference,
    scientific = TRUE,
    digits = 6
  ),
  "\n"
)


cat(
  "Testing-set scaling parameters derived from training set: Yes\n"
)


cat(
  "Testing-set information used to derive scaling parameters: No\n"
)


cat("============================================================\n")