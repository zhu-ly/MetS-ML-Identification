# ==============================================================================
# 03_threshold_sensitivity.R
#
# Sensitivity analysis:
# Classification performance across alternative CatBoost probability thresholds
#
# Objectives:
# 1. Use the locked final CatBoost model without retraining.
# 2. Evaluate the same predefined thresholds in the independent testing set
#    and external validation cohort.
# 3. Use the CatBoost Youden threshold determined from the training set only.
# 4. Illustrate sensitivity/specificity and false-positive/false-negative
#    trade-offs.
#
# Thresholds:
#   0.15
#   0.20
#   Training-set Youden threshold
#   0.30
#   0.35
#
# IMPORTANT:
# - The model is NOT retrained.
# - No threshold is optimized in the testing set.
# - No threshold is optimized in the external validation cohort.
# - No patient-level predictions are written to disk.
# ==============================================================================


library(dplyr)
library(catboost)


source("RR/main/00_setup.R")


set.seed(SEED)


# ==============================================================================
# 1. Common definitions
# ==============================================================================


final_vars <- c(
  "age",
  "BMI",
  "pulse",
  "BUN",
  "TC",
  "UA",
  "HbA1c"
)


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
  
  
  if (
    all(
      unique_values %in% c(
        "No",
        "Yes"
      )
    )
  ) {
    
    return(
      ifelse(
        x_char == "Yes",
        1L,
        0L
      )
    )
  }
  
  
  stop(
    "Unrecognized outcome coding."
  )
}


# ==============================================================================
# 2. Load independent testing set
# ==============================================================================


test_set <- readRDS(
  file.path(
    processed_dir,
    "test_set.rds"
  )
)


required_test_vars <- c(
  final_vars,
  "met_diagnosis"
)


missing_test_vars <- setdiff(
  required_test_vars,
  names(test_set)
)


if (
  length(missing_test_vars) > 0
) {
  
  stop(
    paste(
      "Testing set is missing:",
      paste(
        missing_test_vars,
        collapse = ", "
      )
    )
  )
}


test_y <- to_binary(
  test_set$met_diagnosis
)


stopifnot(
  nrow(test_set) == 2960,
  sum(test_y) == 659,
  !anyNA(
    test_set[
      ,
      required_test_vars
    ]
  )
)


# ==============================================================================
# 3. Load external validation model data
# ==============================================================================


external_file <- file.path(
  processed_dir,
  "external_validation_model_data.rds"
)


if (
  !file.exists(external_file)
) {
  
  stop(
    paste0(
      "Cannot find ",
      external_file,
      ". Run RR/main/08_prepare_external_validation.R first."
    )
  )
}


external_data <- readRDS(
  external_file
)


required_external_vars <- c(
  final_vars,
  "met_diagnosis"
)


missing_external_vars <- setdiff(
  required_external_vars,
  names(external_data)
)


if (
  length(missing_external_vars) > 0
) {
  
  stop(
    paste(
      "External validation dataset is missing:",
      paste(
        missing_external_vars,
        collapse = ", "
      )
    )
  )
}


external_y <- to_binary(
  external_data$met_diagnosis
)


stopifnot(
  nrow(external_data) == 553,
  sum(external_y) == 213,
  !anyNA(
    external_data[
      ,
      required_external_vars
    ]
  )
)


# ==============================================================================
# 4. Load locked CatBoost model
# ==============================================================================


catboost_model_file <- file.path(
  model_dir,
  "locked",
  "model_cat.rds"
)


if (
  !file.exists(catboost_model_file)
) {
  
  stop(
    paste0(
      "Cannot find locked CatBoost model: ",
      catboost_model_file
    )
  )
}


model_cat <- readRDS(
  catboost_model_file
)


# ==============================================================================
# 5. Generate testing-set probabilities using locked CatBoost model
# ==============================================================================


test_pool <- catboost.load_pool(
  data = as.matrix(
    test_set[
      ,
      final_vars
    ]
  )
)


test_prob <- as.numeric(
  catboost.predict(
    model_cat,
    test_pool,
    prediction_type = "Probability"
  )
)


stopifnot(
  length(test_prob) == nrow(test_set),
  all(
    test_prob >= 0 &
      test_prob <= 1
  )
)


# ==============================================================================
# 6. Generate external-validation probabilities using locked CatBoost model
# ==============================================================================


external_pool <- catboost.load_pool(
  data = as.matrix(
    external_data[
      ,
      final_vars
    ]
  )
)


external_prob <- as.numeric(
  catboost.predict(
    model_cat,
    external_pool,
    prediction_type = "Probability"
  )
)


stopifnot(
  length(external_prob) == nrow(external_data),
  all(
    external_prob >= 0 &
      external_prob <= 1
  )
)


# ==============================================================================
# 7. Load training-set CatBoost Youden threshold
# ==============================================================================


threshold_file <- file.path(
  results_dir,
  "locked_train_thresholds.rds"
)


if (
  !file.exists(threshold_file)
) {
  
  stop(
    paste0(
      "Cannot find ",
      threshold_file,
      ". Run RR/main/06_evaluate_models.R first."
    )
  )
}


train_thresholds <- readRDS(
  threshold_file
)


if (
  !"CAT" %in% names(train_thresholds)
) {
  
  stop(
    "CAT threshold is missing from locked_train_thresholds.rds."
  )
}


youden_threshold <- as.numeric(
  train_thresholds$CAT
)


stopifnot(
  length(youden_threshold) == 1,
  is.finite(youden_threshold),
  youden_threshold > 0,
  youden_threshold < 1
)


# Expected locked training-set CatBoost Youden threshold

stopifnot(
  abs(
    youden_threshold -
      0.2498348
  ) < 1e-6
)


# ==============================================================================
# 8. Define thresholds
# ==============================================================================


threshold_table <- data.frame(
  Threshold_Label = c(
    "0.15",
    "0.20",
    "Training-set Youden",
    "0.30",
    "0.35"
  ),
  Threshold = c(
    0.15,
    0.20,
    youden_threshold,
    0.30,
    0.35
  ),
  stringsAsFactors = FALSE
)


# ==============================================================================
# 9. Threshold-performance function
# ==============================================================================


evaluate_threshold <- function(
    probs,
    labels,
    dataset_name,
    threshold_label,
    threshold
) {
  
  pred <- ifelse(
    probs >= threshold,
    1L,
    0L
  )
  
  
  TP <- sum(
    pred == 1 &
      labels == 1
  )
  
  
  FP <- sum(
    pred == 1 &
      labels == 0
  )
  
  
  TN <- sum(
    pred == 0 &
      labels == 0
  )
  
  
  FN <- sum(
    pred == 0 &
      labels == 1
  )
  
  
  N <- length(labels)
  
  events <- sum(
    labels == 1
  )
  
  non_events <- sum(
    labels == 0
  )
  
  
  sensitivity <- ifelse(
    TP + FN > 0,
    TP / (
      TP + FN
    ),
    NA_real_
  )
  
  
  specificity <- ifelse(
    TN + FP > 0,
    TN / (
      TN + FP
    ),
    NA_real_
  )
  
  
  ppv <- ifelse(
    TP + FP > 0,
    TP / (
      TP + FP
    ),
    NA_real_
  )
  
  
  npv <- ifelse(
    TN + FN > 0,
    TN / (
      TN + FN
    ),
    NA_real_
  )
  
  
  accuracy <- (
    TP + TN
  ) / N
  
  
  f1 <- ifelse(
    2 * TP + FP + FN > 0,
    2 * TP / (
      2 * TP +
        FP +
        FN
    ),
    NA_real_
  )
  
  
  false_positive_rate <- ifelse(
    non_events > 0,
    FP / non_events,
    NA_real_
  )
  
  
  false_negative_rate <- ifelse(
    events > 0,
    FN / events,
    NA_real_
  )
  
  
  predicted_positive_rate <- (
    TP + FP
  ) / N
  
  
  data.frame(
    Dataset = dataset_name,
    Threshold_Label = threshold_label,
    Threshold = threshold,
    N = N,
    MetS_Events = events,
    Prevalence = events / N,
    TP = TP,
    FP = FP,
    TN = TN,
    FN = FN,
    Accuracy = accuracy,
    Sensitivity = sensitivity,
    Specificity = specificity,
    PPV = ppv,
    NPV = npv,
    F1_Score = f1,
    False_Positive_Rate = false_positive_rate,
    False_Negative_Rate = false_negative_rate,
    Predicted_Positive_Rate = predicted_positive_rate,
    stringsAsFactors = FALSE
  )
}


# ==============================================================================
# 10. Evaluate thresholds in testing set
# ==============================================================================


test_results_list <- vector(
  "list",
  nrow(threshold_table)
)


for (
  i in seq_len(
    nrow(threshold_table)
  )
) {
  
  test_results_list[[i]] <- evaluate_threshold(
    probs = test_prob,
    labels = test_y,
    dataset_name = "Testing set",
    threshold_label =
      threshold_table$Threshold_Label[i],
    threshold =
      threshold_table$Threshold[i]
  )
}


test_results <- dplyr::bind_rows(
  test_results_list
)


# ==============================================================================
# 11. Evaluate identical thresholds in external validation cohort
# ==============================================================================


external_results_list <- vector(
  "list",
  nrow(threshold_table)
)


for (
  i in seq_len(
    nrow(threshold_table)
  )
) {
  
  external_results_list[[i]] <- evaluate_threshold(
    probs = external_prob,
    labels = external_y,
    dataset_name = "External validation",
    threshold_label =
      threshold_table$Threshold_Label[i],
    threshold =
      threshold_table$Threshold[i]
  )
}


external_results <- dplyr::bind_rows(
  external_results_list
)


# ==============================================================================
# 12. Combine results
# ==============================================================================


combined_results <- dplyr::bind_rows(
  test_results,
  external_results
)


# ==============================================================================
# 13. Reproducibility checks against Supplementary Table S10
# ==============================================================================


expected_results <- data.frame(
  Dataset = c(
    rep(
      "Testing set",
      5
    ),
    rep(
      "External validation",
      5
    )
  ),
  
  Threshold_Label = rep(
    c(
      "0.15",
      "0.20",
      "Training-set Youden",
      "0.30",
      "0.35"
    ),
    2
  ),
  
  Accuracy = c(
    0.668,
    0.711,
    0.749,
    0.783,
    0.803,
    0.584,
    0.633,
    0.667,
    0.700,
    0.732
  ),
  
  Sensitivity = c(
    0.866,
    0.804,
    0.737,
    0.671,
    0.572,
    0.967,
    0.944,
    0.930,
    0.850,
    0.798
  ),
  
  Specificity = c(
    0.611,
    0.684,
    0.752,
    0.816,
    0.869,
    0.344,
    0.438,
    0.503,
    0.606,
    0.691
  ),
  
  PPV = c(
    0.389,
    0.422,
    0.460,
    0.510,
    0.555,
    0.480,
    0.513,
    0.540,
    0.575,
    0.618
  ),
  
  NPV = c(
    0.941,
    0.924,
    0.909,
    0.896,
    0.876,
    0.944,
    0.925,
    0.919,
    0.866,
    0.845
  ),
  
  F1_Score = c(
    0.537,
    0.553,
    0.566,
    0.580,
    0.564,
    0.642,
    0.664,
    0.683,
    0.686,
    0.697
  ),
  
  False_Positive_Rate = c(
    0.389,
    0.316,
    0.248,
    0.184,
    0.131,
    0.656,
    0.562,
    0.497,
    0.394,
    0.309
  ),
  
  False_Negative_Rate = c(
    0.134,
    0.196,
    0.263,
    0.329,
    0.428,
    0.033,
    0.056,
    0.070,
    0.150,
    0.202
  ),
  
  stringsAsFactors = FALSE
)


metrics_to_check <- c(
  "Accuracy",
  "Sensitivity",
  "Specificity",
  "PPV",
  "NPV",
  "F1_Score",
  "False_Positive_Rate",
  "False_Negative_Rate"
)


for (
  metric in metrics_to_check
) {
  
  observed_rounded <- round(
    combined_results[[metric]],
    3
  )
  
  
  expected_value <- expected_results[[metric]]
  
  
  if (
    !all(
      observed_rounded ==
      expected_value
    )
  ) {
    
    stop(
      paste0(
        "Reproducibility check failed for ",
        metric,
        "."
      )
    )
  }
}


# ==============================================================================
# 14. Save aggregate outputs only
# ==============================================================================


write.csv(
  combined_results,
  file.path(
    results_dir,
    "CatBoost_Threshold_Sensitivity.csv"
  ),
  row.names = FALSE
)


formatted_results <- combined_results %>%
  mutate(
    Threshold = ifelse(
      Threshold_Label ==
        "Training-set Youden",
      sprintf(
        "%.4f",
        Threshold
      ),
      sprintf(
        "%.2f",
        Threshold
      )
    ),
    
    Accuracy = sprintf(
      "%.3f",
      Accuracy
    ),
    
    Sensitivity = sprintf(
      "%.3f",
      Sensitivity
    ),
    
    Specificity = sprintf(
      "%.3f",
      Specificity
    ),
    
    PPV = sprintf(
      "%.3f",
      PPV
    ),
    
    NPV = sprintf(
      "%.3f",
      NPV
    ),
    
    F1 = sprintf(
      "%.3f",
      F1_Score
    ),
    
    `False-positive rate` = sprintf(
      "%.3f",
      False_Positive_Rate
    ),
    
    `False-negative rate` = sprintf(
      "%.3f",
      False_Negative_Rate
    )
  ) %>%
  select(
    Dataset,
    Threshold_Label,
    Threshold,
    Accuracy,
    Sensitivity,
    Specificity,
    PPV,
    NPV,
    F1,
    `False-positive rate`,
    `False-negative rate`
  )


write.csv(
  formatted_results,
  file.path(
    results_dir,
    "CatBoost_Threshold_Sensitivity_Formatted.csv"
  ),
  row.names = FALSE
)


# ==============================================================================
# 15. Console summary
# ==============================================================================


cat("\n")
cat("============================================================\n")
cat("CatBoost threshold sensitivity analysis\n")
cat("============================================================\n")


cat(
  "Testing N =",
  length(test_y),
  "\n"
)


cat(
  "Testing MetS cases =",
  sum(test_y),
  "\n"
)


cat(
  "External N =",
  length(external_y),
  "\n"
)


cat(
  "External MetS cases =",
  sum(external_y),
  "\n"
)


cat(
  "Training-set Youden threshold =",
  sprintf(
    "%.7f",
    youden_threshold
  ),
  "\n"
)


cat("\n")
cat("Threshold performance\n")
cat("---------------------\n")


print(
  as.data.frame(
    formatted_results
  ),
  row.names = FALSE
)


cat("\n")
cat("Reproducibility checks\n")
cat("----------------------\n")


cat(
  "Locked CatBoost model used: Yes\n"
)


cat(
  "Model retrained: No\n"
)


cat(
  "Testing-set threshold optimization: No\n"
)


cat(
  "External threshold optimization: No\n"
)


cat(
  "Patient-level predictions written to disk: No\n"
)


cat(
  "Supplementary Table S10 reproduced: Yes\n"
)


cat("============================================================\n")