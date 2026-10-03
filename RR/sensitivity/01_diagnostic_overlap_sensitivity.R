# ==============================================================================
# 01_diagnostic_overlap_sensitivity.R
#
# Diagnostic-overlap sensitivity analysis
#
# Objectives:
# 1. Quantify correlations between the seven retained predictors and the six
#    MetS diagnostic components excluded before formal feature selection.
# 2. Examine BMI-waist and HbA1c-FPG correlations specifically.
# 3. Refit CatBoost models after removing BMI and/or HbA1c while keeping the
#    hyperparameters of the original locked CatBoost model unchanged.
# 4. Compare reduced-variable models with the original seven-variable model
#    using paired DeLong tests in the independent testing set.
#
# IMPORTANT:
# - Diagnostic components are NOT added to the prediction model.
# - Reduced-variable models are refitted only for sensitivity analysis.
# - No additional hyperparameter tuning is performed.
# ==============================================================================


library(dplyr)
library(catboost)
library(pROC)


source("RR/main/00_setup.R")


set.seed(SEED)


# ==============================================================================
# 1. Load development datasets and locked CatBoost model
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


model_cat <- readRDS(
  file.path(
    model_dir,
    "locked",
    "model_cat.rds"
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


diagnostic_vars <- c(
  "waist",
  "SBP",
  "DBP",
  "HDL_C",
  "TG",
  "FPG"
)


required_train_vars <- c(
  final_vars,
  diagnostic_vars,
  "met_diagnosis"
)


required_test_vars <- c(
  final_vars,
  "met_diagnosis"
)


missing_train <- setdiff(
  required_train_vars,
  names(train_set)
)


missing_test <- setdiff(
  required_test_vars,
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
  
  if (is.factor(x)) {
    x <- as.character(x)
  }
  
  if (is.character(x)) {
    
    if (all(na.omit(x) %in% c("0", "1"))) {
      return(as.numeric(x))
    }
    
    if (all(na.omit(x) %in% c("Control", "Case"))) {
      return(
        ifelse(
          x == "Case",
          1,
          0
        )
      )
    }
  }
  
  as.numeric(x)
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
  sum(train_y == 1) == 1538,
  sum(test_y == 1) == 659
)


# ==============================================================================
# 3. Correlations between retained predictors and diagnostic components
# ==============================================================================


cor_data <- train_set %>%
  dplyr::select(
    all_of(
      c(
        final_vars,
        diagnostic_vars
      )
    )
  )


spearman_full <- cor(
  cor_data,
  method = "spearman",
  use = "pairwise.complete.obs"
)


spearman_matrix <- spearman_full[
  final_vars,
  diagnostic_vars,
  drop = FALSE
]


pearson_full <- cor(
  cor_data,
  method = "pearson",
  use = "pairwise.complete.obs"
)


pearson_matrix <- pearson_full[
  final_vars,
  diagnostic_vars,
  drop = FALSE
]


cor_long <- expand.grid(
  Predictor = final_vars,
  Diagnostic_Component = diagnostic_vars,
  stringsAsFactors = FALSE
) %>%
  mutate(
    Spearman_r = mapply(
      function(x, y) {
        spearman_matrix[x, y]
      },
      Predictor,
      Diagnostic_Component
    ),
    Pearson_r = mapply(
      function(x, y) {
        pearson_matrix[x, y]
      },
      Predictor,
      Diagnostic_Component
    ),
    Abs_Spearman_r = abs(Spearman_r)
  ) %>%
  arrange(
    desc(Abs_Spearman_r)
  )


key_pairs <- data.frame(
  Predictor = c(
    "BMI",
    "HbA1c"
  ),
  Diagnostic_Component = c(
    "waist",
    "FPG"
  ),
  stringsAsFactors = FALSE
) %>%
  mutate(
    Spearman_r = mapply(
      function(x, y) {
        spearman_matrix[x, y]
      },
      Predictor,
      Diagnostic_Component
    ),
    Pearson_r = mapply(
      function(x, y) {
        pearson_matrix[x, y]
      },
      Predictor,
      Diagnostic_Component
    )
  )


# ==============================================================================
# 4. Save correlation results
# ==============================================================================


write.csv(
  spearman_matrix,
  file.path(
    results_dir,
    "Correlation_Retained_vs_Diagnostic_Spearman.csv"
  ),
  row.names = TRUE
)


write.csv(
  pearson_matrix,
  file.path(
    results_dir,
    "Correlation_Retained_vs_Diagnostic_Pearson.csv"
  ),
  row.names = TRUE
)


write.csv(
  cor_long,
  file.path(
    results_dir,
    "Correlation_Retained_vs_Diagnostic_Long.csv"
  ),
  row.names = FALSE
)


write.csv(
  key_pairs,
  file.path(
    results_dir,
    "Correlation_Key_Pairs.csv"
  ),
  row.names = FALSE
)


# ==============================================================================
# 5. Read hyperparameters from the locked CatBoost model
# ==============================================================================


cat_params <- catboost.get_model_params(
  model_cat
)


flat_params <- cat_params$flat_params


if (is.null(flat_params)) {
  stop(
    paste0(
      "Unable to retrieve flat_params from the locked ",
      "CatBoost model."
    )
  )
}


original_iterations <- as.integer(
  flat_params$iterations
)


original_learning_rate <- as.numeric(
  flat_params$learning_rate
)


original_depth <- as.integer(
  flat_params$depth
)


original_l2_leaf_reg <- as.numeric(
  flat_params$l2_leaf_reg
)


original_border_count <- as.integer(
  flat_params$border_count
)


original_random_seed <- as.integer(
  flat_params$random_seed
)


# Verify the locked parameters used in the manuscript
stopifnot(
  original_iterations == 73,
  abs(original_learning_rate - 0.1) < 1e-12,
  original_depth == 4,
  abs(original_l2_leaf_reg - 5) < 1e-12,
  original_border_count == 254,
  original_random_seed == 123
)


# ==============================================================================
# 6. Define original and reduced predictor sets
# ==============================================================================


model_sets <- list(
  
  Original_7_variables = c(
    "age",
    "BMI",
    "pulse",
    "BUN",
    "TC",
    "UA",
    "HbA1c"
  ),
  
  Without_BMI = c(
    "age",
    "pulse",
    "BUN",
    "TC",
    "UA",
    "HbA1c"
  ),
  
  Without_HbA1c = c(
    "age",
    "BMI",
    "pulse",
    "BUN",
    "TC",
    "UA"
  ),
  
  Without_BMI_and_HbA1c = c(
    "age",
    "pulse",
    "BUN",
    "TC",
    "UA"
  )
)


# ==============================================================================
# 7. Obtain predictions from the original locked seven-variable model
# ==============================================================================


test_original_pool <- catboost.load_pool(
  data = as.matrix(
    test_set[
      ,
      final_vars
    ]
  )
)


pred_original <- catboost.predict(
  model_cat,
  test_original_pool,
  prediction_type = "Probability"
)


roc_original <- pROC::roc(
  response = test_y,
  predictor = pred_original,
  levels = c(0, 1),
  direction = "<",
  quiet = TRUE
)


auc_original <- as.numeric(
  pROC::auc(
    roc_original
  )
)


ci_original <- as.numeric(
  pROC::ci.auc(
    roc_original,
    method = "delong"
  )
)


brier_original <- mean(
  (pred_original - test_y)^2
)


# ==============================================================================
# 8. Refit reduced-variable CatBoost models
#
# The original CatBoost hyperparameters are retained unchanged.
# No additional hyperparameter tuning is performed.
# ==============================================================================


results_list <- list()


results_list[["Original_7_variables"]] <- data.frame(
  Model = "Original 7-variable model",
  N_Predictors = 7,
  AUC = auc_original,
  CI_Lower = ci_original[1],
  CI_Upper = ci_original[3],
  Brier = brier_original,
  stringsAsFactors = FALSE
)


pred_list <- list(
  Original_7_variables = pred_original
)


roc_list <- list(
  Original_7_variables = roc_original
)


sensitivity_names <- c(
  "Without_BMI",
  "Without_HbA1c",
  "Without_BMI_and_HbA1c"
)


fixed_params <- list(
  iterations = original_iterations,
  learning_rate = original_learning_rate,
  depth = original_depth,
  l2_leaf_reg = original_l2_leaf_reg,
  loss_function = "Logloss",
  random_seed = original_random_seed,
  border_count = original_border_count,
  logging_level = "Silent"
)


for (model_name in sensitivity_names) {
  
  vars <- model_sets[[model_name]]
  
  cat(
    "Refitting sensitivity model:",
    model_name,
    "\n"
  )
  
  
  train_pool_temp <- catboost.load_pool(
    data = as.matrix(
      train_set[
        ,
        vars
      ]
    ),
    label = train_y
  )
  
  
  test_pool_temp <- catboost.load_pool(
    data = as.matrix(
      test_set[
        ,
        vars
      ]
    )
  )
  
  
  model_temp <- catboost.train(
    train_pool_temp,
    params = fixed_params
  )
  
  
  pred_temp <- catboost.predict(
    model_temp,
    test_pool_temp,
    prediction_type = "Probability"
  )
  
  
  roc_temp <- pROC::roc(
    response = test_y,
    predictor = pred_temp,
    levels = c(0, 1),
    direction = "<",
    quiet = TRUE
  )
  
  
  auc_temp <- as.numeric(
    pROC::auc(
      roc_temp
    )
  )
  
  
  ci_temp <- as.numeric(
    pROC::ci.auc(
      roc_temp,
      method = "delong"
    )
  )
  
  
  brier_temp <- mean(
    (pred_temp - test_y)^2
  )
  
  
  model_label <- switch(
    model_name,
    Without_BMI =
      "Without BMI",
    Without_HbA1c =
      "Without HbA1c",
    Without_BMI_and_HbA1c =
      "Without BMI and HbA1c"
  )
  
  
  results_list[[model_name]] <- data.frame(
    Model = model_label,
    N_Predictors = length(vars),
    AUC = auc_temp,
    CI_Lower = ci_temp[1],
    CI_Upper = ci_temp[3],
    Brier = brier_temp,
    stringsAsFactors = FALSE
  )
  
  
  pred_list[[model_name]] <- pred_temp
  roc_list[[model_name]] <- roc_temp
}


performance_table <- dplyr::bind_rows(
  results_list
)


# ==============================================================================
# 9. Paired DeLong comparisons with the original model
# ==============================================================================


delong_results <- list()


for (model_name in sensitivity_names) {
  
  test_delong <- pROC::roc.test(
    roc_original,
    roc_list[[model_name]],
    paired = TRUE,
    method = "delong"
  )
  
  
  reduced_auc <- as.numeric(
    pROC::auc(
      roc_list[[model_name]]
    )
  )
  
  
  model_label <- switch(
    model_name,
    Without_BMI =
      "Without BMI",
    Without_HbA1c =
      "Without HbA1c",
    Without_BMI_and_HbA1c =
      "Without BMI and HbA1c"
  )
  
  
  delong_results[[model_name]] <- data.frame(
    Comparison = paste(
      "Original 7-variable model vs",
      model_label
    ),
    Original_AUC = auc_original,
    Reduced_AUC = reduced_auc,
    
    # Supplementary table reports:
    # reduced model AUC minus original model AUC
    Delta_AUC = reduced_auc - auc_original,
    
    DeLong_P = test_delong$p.value,
    stringsAsFactors = FALSE
  )
}


delong_table <- dplyr::bind_rows(
  delong_results
)


# ==============================================================================
# 10. Combine sensitivity-analysis results
# ==============================================================================


performance_table <- performance_table %>%
  mutate(
    Delta_AUC = AUC - auc_original
  )


performance_table$Delta_AUC[
  performance_table$Model ==
    "Original 7-variable model"
] <- NA_real_


performance_table <- performance_table %>%
  left_join(
    delong_table %>%
      transmute(
        Model = case_when(
          grepl(
            "Without BMI and HbA1c$",
            Comparison
          ) ~
            "Without BMI and HbA1c",
          
          grepl(
            "Without HbA1c$",
            Comparison
          ) ~
            "Without HbA1c",
          
          grepl(
            "Without BMI$",
            Comparison
          ) ~
            "Without BMI",
          
          TRUE ~ NA_character_
        ),
        DeLong_P = DeLong_P
      ),
    by = "Model"
  )


# ==============================================================================
# 11. Save aggregate results
# ==============================================================================


write.csv(
  performance_table,
  file.path(
    results_dir,
    "CatBoost_Sensitivity_Diagnostic_Overlap.csv"
  ),
  row.names = FALSE
)


write.csv(
  delong_table,
  file.path(
    results_dir,
    "CatBoost_Sensitivity_DeLong.csv"
  ),
  row.names = FALSE
)


# ==============================================================================
# 12. Reproducibility checks
# ==============================================================================


# Correlation values reported in the supplementary material
stopifnot(
  abs(
    spearman_matrix["BMI", "waist"] -
      0.839
  ) < 0.001,
  
  abs(
    spearman_matrix["HbA1c", "FPG"] -
      0.418
  ) < 0.001,
  
  abs(
    pearson_matrix["BMI", "waist"] -
      0.817
  ) < 0.001,
  
  abs(
    pearson_matrix["HbA1c", "FPG"] -
      0.757
  ) < 0.001
)


# Original locked-model performance
stopifnot(
  abs(
    auc_original -
      0.829
  ) < 0.001,
  
  abs(
    brier_original -
      0.127
  ) < 0.001
)


# ==============================================================================
# 13. Final console summary
# ==============================================================================


cat("\n")
cat("============================================================\n")
cat("Diagnostic-overlap sensitivity analysis\n")
cat("============================================================\n")

cat(
  "Training N =",
  nrow(train_set),
  "\n"
)

cat(
  "Testing N =",
  nrow(test_set),
  "\n"
)

cat("\n")
cat("Key correlations\n")
cat("----------------\n")

print(
  key_pairs %>%
    mutate(
      Spearman_r = round(
        Spearman_r,
        3
      ),
      Pearson_r = round(
        Pearson_r,
        3
      )
    )
)


cat("\n")
cat("CatBoost sensitivity analysis\n")
cat("-----------------------------\n")


print(
  performance_table %>%
    mutate(
      AUC_95CI = sprintf(
        "%.3f (%.3f-%.3f)",
        AUC,
        CI_Lower,
        CI_Upper
      ),
      
      Brier = round(
        Brier,
        3
      ),
      
      Delta_AUC = round(
        Delta_AUC,
        3
      ),
      
      DeLong_P_display = case_when(
        is.na(DeLong_P) ~
          "Reference",
        
        DeLong_P < 0.001 ~
          "<0.001",
        
        TRUE ~
          sprintf(
            "%.3f",
            DeLong_P
          )
      )
    ) %>%
    select(
      Model,
      N_Predictors,
      AUC_95CI,
      Brier,
      Delta_AUC,
      DeLong_P_display
    )
)


cat("\n")
cat("Locked CatBoost hyperparameters\n")
cat("-------------------------------\n")

cat(
  "iterations =",
  original_iterations,
  "\n"
)

cat(
  "learning_rate =",
  original_learning_rate,
  "\n"
)

cat(
  "depth =",
  original_depth,
  "\n"
)

cat(
  "l2_leaf_reg =",
  original_l2_leaf_reg,
  "\n"
)

cat(
  "border_count =",
  original_border_count,
  "\n"
)

cat(
  "random_seed =",
  original_random_seed,
  "\n"
)


cat("\n")
cat("Additional hyperparameter tuning performed: No\n")
cat("Diagnostic components included as predictors: No\n")

cat("============================================================\n")