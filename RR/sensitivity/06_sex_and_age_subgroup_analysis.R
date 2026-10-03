# ==============================================================================
# 06_sex_and_age_subgroup_analysis.R
#
# Sensitivity and subgroup analyses:
#
# Part A:
#   Compare the original locked 7-predictor CatBoost model with a separately
#   trained 8-predictor CatBoost model that additionally includes sex.
#
# Part B:
#   Evaluate the original locked 7-predictor CatBoost model within predefined
#   sex and age subgroups of the independent testing set.
#
# IMPORTANT:
# - The original locked 7-predictor CatBoost model is NEVER modified.
# - The sex-augmented model is a separate sensitivity-analysis model.
# - The sex-augmented model follows the original CatBoost CV/tuning procedure.
# - Subgroup analyses use predictions from the original locked model only.
# - No model is retrained separately within any subgroup.
# ==============================================================================


library(dplyr)
library(catboost)
library(pROC)


source("RR/main/00_setup.R")


set.seed(SEED)


# ==============================================================================
# 1. Load training and testing sets
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


# ==============================================================================
# 2. Define predictors
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


# Original data variable name is "gender".
# In the manuscript this variable is reported as "sex".
#
# Coding:
#   0 = Male
#   1 = Female

sex_vars <- c(
  "age",
  "gender",
  "BMI",
  "pulse",
  "BUN",
  "TC",
  "UA",
  "HbA1c"
)


required_vars <- unique(
  c(
    sex_vars,
    "met_diagnosis"
  )
)


missing_train_vars <- setdiff(
  required_vars,
  names(train_set)
)


missing_test_vars <- setdiff(
  required_vars,
  names(test_set)
)


if (
  length(missing_train_vars) > 0
) {
  
  stop(
    paste(
      "Training set is missing:",
      paste(
        missing_train_vars,
        collapse = ", "
      )
    )
  )
}


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


if (
  anyNA(
    train_set[
      ,
      required_vars
    ]
  )
) {
  
  stop(
    "Training set contains missing values in variables required for this analysis."
  )
}


if (
  anyNA(
    test_set[
      ,
      required_vars
    ]
  )
) {
  
  stop(
    "Testing set contains missing values in variables required for this analysis."
  )
}


# ==============================================================================
# 3. Outcome conversion
# ==============================================================================


to_binary <- function(x) {
  
  x_char <- as.character(x)
  
  unique_values <- unique(
    na.omit(
      x_char
    )
  )
  
  
  if (
    all(
      unique_values %in% c(
        "0",
        "1"
      )
    )
  ) {
    
    return(
      as.integer(
        x_char
      )
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


train_y <- to_binary(
  train_set$met_diagnosis
)


test_y <- to_binary(
  test_set$met_diagnosis
)


stopifnot(
  nrow(train_set) == 6907,
  sum(train_y) == 1538,
  nrow(test_set) == 2960,
  sum(test_y) == 659
)


# ==============================================================================
# 4. Verify sex coding
# ==============================================================================


if (
  !all(
    unique(
      train_set$gender
    ) %in% c(
      0,
      1
    )
  )
) {
  
  stop(
    "Unexpected gender coding in training set."
  )
}


if (
  !all(
    unique(
      test_set$gender
    ) %in% c(
      0,
      1
    )
  )
) {
  
  stop(
    "Unexpected gender coding in testing set."
  )
}


cat("\n")
cat("Training-set sex counts\n")
cat("-----------------------\n")

print(
  table(
    train_set$gender
  )
)


cat("\n")
cat("Testing-set sex counts\n")
cat("----------------------\n")

print(
  table(
    test_set$gender
  )
)


# ==============================================================================
# 5. Load original locked 7-predictor CatBoost model
# ==============================================================================


original_model_file <- file.path(
  model_dir,
  "locked",
  "model_cat.rds"
)


if (
  !file.exists(
    original_model_file
  )
) {
  
  stop(
    paste0(
      "Cannot find locked CatBoost model: ",
      original_model_file
    )
  )
}


model_cat_original <- readRDS(
  original_model_file
)


# ==============================================================================
# 6. Original locked-model predictions in independent testing set
# ==============================================================================


test_pool_original <- catboost.load_pool(
  data = as.matrix(
    test_set[
      ,
      final_vars
    ]
  )
)


prob_original <- as.numeric(
  catboost.predict(
    model_cat_original,
    test_pool_original,
    prediction_type = "Probability"
  )
)


stopifnot(
  length(prob_original) ==
    nrow(test_set),
  all(
    prob_original >= 0 &
      prob_original <= 1
  )
)


# ==============================================================================
# 7. Train separate CatBoost + sex sensitivity model
#
# This reproduces the original sensitivity-analysis procedure:
#
# Step 1:
#   baseline 10-fold CV
#
# Step 2:
#   grid search:
#       depth = 4, 6
#       l2_leaf_reg = 1, 3, 5
#
# Step 3:
#   final refit using:
#       selected iteration × 1.10
#
# Fixed settings:
#   learning_rate = 0.1
#   border_count = 254
#   loss_function = Logloss
#   seed = 123
#
# Testing-set information is NOT used for model tuning.
# ==============================================================================


cat("\n")
cat("============================================================\n")
cat("Training separate CatBoost + sex sensitivity model\n")
cat("============================================================\n")


train_x_sex <- as.matrix(
  train_set[
    ,
    sex_vars
  ]
)


train_pool_sex <- catboost.load_pool(
  data = train_x_sex,
  label = train_y
)


# ------------------------------------------------------------------------------
# Step 1. Baseline cross-validation
# ------------------------------------------------------------------------------


cat(
  "Step 1/3: baseline 10-fold cross-validation...\n"
)


set.seed(SEED)


cat_cv_base_sex <- catboost.cv(
  
  train_pool_sex,
  
  params = list(
    
    iterations = 1000,
    
    learning_rate = 0.1,
    
    depth = 6,
    
    l2_leaf_reg = 3,
    
    loss_function = "Logloss",
    
    eval_metric = "AUC",
    
    random_seed = SEED,
    
    border_count = 254,
    
    logging_level = "Silent",
    
    allow_writing_files = FALSE
  ),
  
  fold_count = 10,
  
  early_stopping_rounds = 20
)


best_base_iter_sex <- which.max(
  cat_cv_base_sex$test.AUC.mean
)


cat(
  "Baseline best iteration =",
  best_base_iter_sex,
  "\n"
)


# ------------------------------------------------------------------------------
# Step 2. Grid search
# ------------------------------------------------------------------------------


cat(
  "Step 2/3: CatBoost grid search...\n"
)


cat_grid <- expand.grid(
  
  depth = c(
    4,
    6
  ),
  
  l2_leaf_reg = c(
    1,
    3,
    5
  )
)


cat_results_sex <- data.frame()


for (
  i in seq_len(
    nrow(cat_grid)
  )
) {
  
  cat(
    "  Testing depth =",
    cat_grid$depth[i],
    ", l2_leaf_reg =",
    cat_grid$l2_leaf_reg[i],
    "\n"
  )
  
  
  set.seed(SEED)
  
  
  cv <- catboost.cv(
    
    train_pool_sex,
    
    params = list(
      
      iterations = max(
        10,
        round(
          best_base_iter_sex *
            1.5
        )
      ),
      
      learning_rate = 0.1,
      
      depth =
        cat_grid$depth[i],
      
      l2_leaf_reg =
        cat_grid$l2_leaf_reg[i],
      
      loss_function =
        "Logloss",
      
      eval_metric =
        "AUC",
      
      random_seed =
        SEED,
      
      border_count =
        254,
      
      logging_level =
        "Silent",
      
      allow_writing_files =
        FALSE
    ),
    
    fold_count = 10,
    
    early_stopping_rounds = 20
  )
  
  
  best_iter_i <- which.max(
    cv$test.AUC.mean
  )
  
  
  best_auc_i <- max(
    cv$test.AUC.mean
  )
  
  
  cat_results_sex <- rbind(
    
    cat_results_sex,
    
    data.frame(
      
      Grid_Index = i,
      
      depth =
        cat_grid$depth[i],
      
      l2_leaf_reg =
        cat_grid$l2_leaf_reg[i],
      
      CV_AUC =
        best_auc_i,
      
      CV_Best_Iteration =
        best_iter_i
    )
  )
}


best_cat_sex <- cat_results_sex[
  which.max(
    cat_results_sex$CV_AUC
  ),
  ,
  drop = FALSE
]


final_sex_iterations <- max(
  1,
  round(
    best_cat_sex$CV_Best_Iteration *
      1.1
  )
)


cat("\n")
cat("Selected CatBoost + sex parameters\n")
cat("----------------------------------\n")


print(
  best_cat_sex,
  row.names = FALSE
)


cat(
  "Final refit iterations =",
  final_sex_iterations,
  "\n"
)


# ------------------------------------------------------------------------------
# Step 3. Final refit
# ------------------------------------------------------------------------------


cat(
  "Step 3/3: fitting final CatBoost + sex sensitivity model...\n"
)


set.seed(SEED)


model_cat_sex <- catboost.train(
  
  train_pool_sex,
  
  params = list(
    
    iterations =
      final_sex_iterations,
    
    learning_rate =
      0.1,
    
    depth =
      best_cat_sex$depth,
    
    l2_leaf_reg =
      best_cat_sex$l2_leaf_reg,
    
    loss_function =
      "Logloss",
    
    random_seed =
      SEED,
    
    border_count =
      254,
    
    logging_level =
      "Silent",
    
    allow_writing_files =
      FALSE
  )
)


# ==============================================================================
# 8. Predict with CatBoost + sex in independent testing set
# ==============================================================================


test_pool_sex <- catboost.load_pool(
  data = as.matrix(
    test_set[
      ,
      sex_vars
    ]
  )
)


prob_sex <- as.numeric(
  catboost.predict(
    model_cat_sex,
    test_pool_sex,
    prediction_type = "Probability"
  )
)


stopifnot(
  length(prob_sex) ==
    nrow(test_set),
  all(
    prob_sex >= 0 &
      prob_sex <= 1
  )
)


# ==============================================================================
# 9. Compare original model vs CatBoost + sex
# ==============================================================================


roc_original <- pROC::roc(
  response = test_y,
  predictor = prob_original,
  quiet = TRUE,
  direction = "<"
)


roc_sex <- pROC::roc(
  response = test_y,
  predictor = prob_sex,
  quiet = TRUE,
  direction = "<"
)


auc_original <- as.numeric(
  pROC::auc(
    roc_original
  )
)


auc_sex <- as.numeric(
  pROC::auc(
    roc_sex
  )
)


ci_original <- pROC::ci.auc(
  roc_original
)


ci_sex <- pROC::ci.auc(
  roc_sex
)


brier_original <- mean(
  (
    prob_original -
      test_y
  )^2
)


brier_sex <- mean(
  (
    prob_sex -
      test_y
  )^2
)


delong_result <- pROC::roc.test(
  
  roc_original,
  
  roc_sex,
  
  method = "delong",
  
  paired = TRUE
)


sensitivity_table <- data.frame(
  
  Model = c(
    "Original CatBoost (7 predictors)",
    "CatBoost + sex (8 predictors)"
  ),
  
  AUC = c(
    auc_original,
    auc_sex
  ),
  
  CI_Lower = c(
    as.numeric(
      ci_original[1]
    ),
    as.numeric(
      ci_sex[1]
    )
  ),
  
  CI_Upper = c(
    as.numeric(
      ci_original[3]
    ),
    as.numeric(
      ci_sex[3]
    )
  ),
  
  Brier = c(
    brier_original,
    brier_sex
  ),
  
  stringsAsFactors = FALSE
)


delong_table <- data.frame(
  
  Comparison =
    "Original CatBoost vs CatBoost + sex",
  
  P_Value =
    as.numeric(
      delong_result$p.value
    ),
  
  stringsAsFactors = FALSE
)


# ==============================================================================
# 10. Reproducibility checks against current Supplementary Table S2
# ==============================================================================


expected_s2 <- data.frame(
  
  Model = c(
    "Original CatBoost (7 predictors)",
    "CatBoost + sex (8 predictors)"
  ),
  
  AUC = c(
    0.829,
    0.833
  ),
  
  CI_Lower = c(
    0.813,
    0.816
  ),
  
  CI_Upper = c(
    0.846,
    0.850
  ),
  
  Brier = c(
    0.127,
    0.126
  )
)


metrics_s2 <- c(
  "AUC",
  "CI_Lower",
  "CI_Upper",
  "Brier"
)


for (
  metric in metrics_s2
) {
  
  if (
    !all(
      round(
        sensitivity_table[[metric]],
        3
      ) ==
      expected_s2[[metric]]
    )
  ) {
    
    stop(
      paste0(
        "Supplementary Table S2 check failed for ",
        metric,
        "."
      )
    )
  }
}


if (
  round(
    delong_result$p.value,
    3
  ) !=
  0.007
) {
  
  stop(
    paste0(
      "Supplementary Table S2 DeLong check failed. Observed P = ",
      format(
        delong_result$p.value,
        digits = 8
      ),
      "."
    )
  )
}


# ==============================================================================
# 11. Prepare original-model testing predictions for subgroup analyses
# ==============================================================================


test_eval <- test_set %>%
  mutate(
    
    outcome =
      test_y,
    
    pred_prob =
      prob_original,
    
    sex = case_when(
      
      gender == 0 ~
        "Male",
      
      gender == 1 ~
        "Female",
      
      TRUE ~
        NA_character_
    ),
    
    age_group = case_when(
      
      age >= 45 &
        age < 60 ~
        "45-59 years",
      
      age >= 60 ~
        "≥60 years",
      
      TRUE ~
        NA_character_
    )
  )


if (
  anyNA(
    test_eval$sex
  )
) {
  
  stop(
    "Missing or unexpected sex category in testing set."
  )
}


if (
  anyNA(
    test_eval$age_group
  )
) {
  
  stop(
    "Missing or unexpected age category in testing set."
  )
}


# ==============================================================================
# 12. Subgroup performance function
# ==============================================================================


calc_subgroup_performance <- function(
    data,
    group_name
) {
  
  if (
    nrow(data) == 0
  ) {
    
    stop(
      paste0(
        "Subgroup contains no observations: ",
        group_name
      )
    )
  }
  
  
  if (
    length(
      unique(
        data$outcome
      )
    ) < 2
  ) {
    
    stop(
      paste0(
        "Subgroup contains only one outcome class: ",
        group_name
      )
    )
  }
  
  
  roc_obj <- pROC::roc(
    
    response =
      data$outcome,
    
    predictor =
      data$pred_prob,
    
    quiet =
      TRUE,
    
    direction =
      "<"
  )
  
  
  ci_obj <- pROC::ci.auc(
    roc_obj
  )
  
  
  data.frame(
    
    Subgroup =
      group_name,
    
    N =
      nrow(data),
    
    MetS_cases =
      sum(
        data$outcome == 1
      ),
    
    Prevalence =
      mean(
        data$outcome == 1
      ),
    
    AUC =
      as.numeric(
        pROC::auc(
          roc_obj
        )
      ),
    
    CI_Lower =
      as.numeric(
        ci_obj[1]
      ),
    
    CI_Upper =
      as.numeric(
        ci_obj[3]
      ),
    
    Brier =
      mean(
        (
          data$pred_prob -
            data$outcome
        )^2
      ),
    
    stringsAsFactors = FALSE
  )
}


# ==============================================================================
# 13. Calculate predefined sex and age subgroups
# ==============================================================================


subgroup_results <- dplyr::bind_rows(
  
  calc_subgroup_performance(
    dplyr::filter(
      test_eval,
      sex == "Male"
    ),
    "Male"
  ),
  
  calc_subgroup_performance(
    dplyr::filter(
      test_eval,
      sex == "Female"
    ),
    "Female"
  ),
  
  calc_subgroup_performance(
    dplyr::filter(
      test_eval,
      age_group ==
        "45-59 years"
    ),
    "45-59 years"
  ),
  
  calc_subgroup_performance(
    dplyr::filter(
      test_eval,
      age_group ==
        "≥60 years"
    ),
    "≥60 years"
  )
)


# ==============================================================================
# 14. Reproducibility checks against current Supplementary Table S3
# ==============================================================================


expected_s3 <- data.frame(
  
  Subgroup = c(
    "Male",
    "Female",
    "45-59 years",
    "≥60 years"
  ),
  
  N = c(
    1362,
    1598,
    1445,
    1515
  ),
  
  MetS_cases = c(
    292,
    367,
    308,
    351
  ),
  
  Prevalence = c(
    0.214,
    0.230,
    0.213,
    0.232
  ),
  
  AUC = c(
    0.852,
    0.808,
    0.829,
    0.829
  ),
  
  CI_Lower = c(
    0.829,
    0.784,
    0.806,
    0.806
  ),
  
  CI_Upper = c(
    0.875,
    0.833,
    0.853,
    0.853
  ),
  
  Brier = c(
    0.118,
    0.135,
    0.125,
    0.129
  ),
  
  stringsAsFactors = FALSE
)


if (
  !all(
    subgroup_results$Subgroup ==
    expected_s3$Subgroup
  )
) {
  
  stop(
    "Supplementary Table S3 subgroup order/name check failed."
  )
}


if (
  !all(
    subgroup_results$N ==
    expected_s3$N
  )
) {
  
  stop(
    "Supplementary Table S3 sample-size check failed."
  )
}


if (
  !all(
    subgroup_results$MetS_cases ==
    expected_s3$MetS_cases
  )
) {
  
  stop(
    "Supplementary Table S3 MetS-case count check failed."
  )
}


metrics_s3 <- c(
  "Prevalence",
  "AUC",
  "CI_Lower",
  "CI_Upper",
  "Brier"
)


for (
  metric in metrics_s3
) {
  
  if (
    !all(
      round(
        subgroup_results[[metric]],
        3
      ) ==
      expected_s3[[metric]]
    )
  ) {
    
    stop(
      paste0(
        "Supplementary Table S3 check failed for ",
        metric,
        "."
      )
    )
  }
}


# ==============================================================================
# 15. Save aggregate outputs and separate sensitivity model
# ==============================================================================


sensitivity_model_dir <- file.path(
  model_dir,
  "sensitivity"
)


dir.create(
  sensitivity_model_dir,
  showWarnings = FALSE,
  recursive = TRUE
)


saveRDS(
  model_cat_sex,
  file.path(
    sensitivity_model_dir,
    "model_cat_plus_sex.rds"
  )
)


write.csv(
  cat_results_sex,
  file.path(
    results_dir,
    "CatBoost_Plus_Sex_Tuning_Results.csv"
  ),
  row.names = FALSE
)


write.csv(
  sensitivity_table,
  file.path(
    results_dir,
    "Sensitivity_Adding_Sex.csv"
  ),
  row.names = FALSE
)


write.csv(
  delong_table,
  file.path(
    results_dir,
    "Sensitivity_Adding_Sex_DeLong.csv"
  ),
  row.names = FALSE
)


write.csv(
  subgroup_results,
  file.path(
    results_dir,
    "CatBoost_Sex_Age_Subgroup_Performance.csv"
  ),
  row.names = FALSE
)


# ==============================================================================
# 16. Formatted Supplementary Tables S2 and S3
# ==============================================================================


formatted_s2 <- data.frame(
  
  Model =
    sensitivity_table$Model,
  
  AUC =
    sprintf(
      "%.3f",
      sensitivity_table$AUC
    ),
  
  `95% CI` =
    paste0(
      sprintf(
        "%.3f",
        sensitivity_table$CI_Lower
      ),
      "–",
      sprintf(
        "%.3f",
        sensitivity_table$CI_Upper
      )
    ),
  
  `Brier score` =
    sprintf(
      "%.3f",
      sensitivity_table$Brier
    ),
  
  `DeLong P value` =
    c(
      "",
      sprintf(
        "%.3f",
        delong_result$p.value
      )
    ),
  
  check.names = FALSE
)


formatted_s3 <- data.frame(
  
  Subgroup =
    subgroup_results$Subgroup,
  
  n =
    subgroup_results$N,
  
  `MetS cases` =
    subgroup_results$MetS_cases,
  
  `Prevalence (%)` =
    sprintf(
      "%.1f",
      100 *
        subgroup_results$Prevalence
    ),
  
  AUC =
    sprintf(
      "%.3f",
      subgroup_results$AUC
    ),
  
  `95% CI` =
    paste0(
      sprintf(
        "%.3f",
        subgroup_results$CI_Lower
      ),
      "–",
      sprintf(
        "%.3f",
        subgroup_results$CI_Upper
      )
    ),
  
  `Brier score` =
    sprintf(
      "%.3f",
      subgroup_results$Brier
    ),
  
  check.names = FALSE
)


write.csv(
  formatted_s2,
  file.path(
    results_dir,
    "Supplementary_Table_S2_Sex_Sensitivity.csv"
  ),
  row.names = FALSE
)


write.csv(
  formatted_s3,
  file.path(
    results_dir,
    "Supplementary_Table_S3_Subgroup_Performance.csv"
  ),
  row.names = FALSE
)


# ==============================================================================
# 17. Console summary
# ==============================================================================


cat("\n\n")
cat("============================================================\n")
cat("Sex sensitivity and subgroup analysis\n")
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
cat("Supplementary Table S2\n")
cat("----------------------\n")


print(
  formatted_s2,
  row.names = FALSE
)


cat("\n")
cat("Supplementary Table S3\n")
cat("----------------------\n")


print(
  formatted_s3,
  row.names = FALSE
)


cat("\n")
cat("CatBoost + sex tuning\n")
cat("---------------------\n")


cat(
  "Selected depth =",
  best_cat_sex$depth,
  "\n"
)


cat(
  "Selected l2_leaf_reg =",
  best_cat_sex$l2_leaf_reg,
  "\n"
)


cat(
  "CV-best iteration =",
  best_cat_sex$CV_Best_Iteration,
  "\n"
)


cat(
  "Final refit iterations =",
  final_sex_iterations,
  "\n"
)


cat("\n")
cat("Reproducibility checks\n")
cat("----------------------\n")


cat(
  "Original locked 7-predictor CatBoost used: Yes\n"
)


cat(
  "Original locked model modified: No\n"
)


cat(
  "Separate CatBoost + sex model trained: Yes\n"
)


cat(
  "Testing set used for tuning: No\n"
)


cat(
  "Subgroup-specific model retraining: No\n"
)


cat(
  "Subgroup predictions from original locked model: Yes\n"
)


cat(
  "Supplementary Table S2 reproduced: Yes\n"
)


cat(
  "Supplementary Table S3 reproduced: Yes\n"
)


cat("============================================================\n")