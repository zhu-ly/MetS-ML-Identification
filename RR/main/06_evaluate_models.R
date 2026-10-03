# ==============================================================================
# 06_evaluate_models.R
# Evaluate the six locked final models in the training and testing sets
# ==============================================================================

library(dplyr)
library(caret)
library(pROC)
library(xgboost)
library(lightgbm)
library(catboost)
library(ResourceSelection)
library(CalibrationCurves)
library(dcurves)
library(ggplot2)

source("RR/main/00_setup.R")


# ==============================================================================
# 1. Paths
# ==============================================================================

locked_model_dir <- file.path(
  model_dir,
  "locked"
)

required_model_files <- c(
  "model_lr.rds",
  "model_lasso.rds",
  "model_rf.rds",
  "model_xgb.rds",
  "model_lgbm.rds",
  "model_cat.rds"
)

missing_model_files <- required_model_files[
  !file.exists(
    file.path(
      locked_model_dir,
      required_model_files
    )
  )
]

if (length(missing_model_files) > 0) {
  stop(
    "Locked final model files are missing: ",
    paste(
      missing_model_files,
      collapse = ", "
    )
  )
}
# ==============================================================================
# 2. Load training and testing datasets
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


stopifnot(
  nrow(train_set) == 6907,
  nrow(test_set) == 2960,
  sum(train_set$met_diagnosis == "1") == 1538,
  sum(test_set$met_diagnosis == "1") == 659
)
# ==============================================================================
# 3. Predictor matrices and binary outcomes
# ==============================================================================

train_x <- as.matrix(
  train_set[, final_predictors]
)

test_x <- as.matrix(
  test_set[, final_predictors]
)


convert_binary <- function(x) {
  
  x_char <- as.character(x)
  
  if (
    all(
      na.omit(
        unique(x_char)
      ) %in% c("0", "1")
    )
  ) {
    
    return(
      as.integer(x_char)
    )
    
  } else if (
    all(
      na.omit(
        unique(x_char)
      ) %in% c("Control", "Case")
    )
  ) {
    
    return(
      ifelse(
        x_char == "Case",
        1,
        0
      )
    )
    
  } else {
    
    stop(
      "Unrecognized met_diagnosis coding."
    )
  }
}


train_y <- convert_binary(
  train_set$met_diagnosis
)

test_y <- convert_binary(
  test_set$met_diagnosis
)
# ==============================================================================
# 4. Load locked final model objects
# ==============================================================================

models <- list(
  
  LR = readRDS(
    file.path(
      locked_model_dir,
      "model_lr.rds"
    )
  ),
  
  LASSO = readRDS(
    file.path(
      locked_model_dir,
      "model_lasso.rds"
    )
  ),
  
  RF = readRDS(
    file.path(
      locked_model_dir,
      "model_rf.rds"
    )
  ),
  
  XGB = readRDS(
    file.path(
      locked_model_dir,
      "model_xgb.rds"
    )
  ),
  
  LGBM = readRDS(
    file.path(
      locked_model_dir,
      "model_lgbm.rds"
    )
  ),
  
  CAT = readRDS(
    file.path(
      locked_model_dir,
      "model_cat.rds"
    )
  )
)
# ==============================================================================
# 5. Unified probability-prediction function
# ==============================================================================

get_probs <- function(
    model_obj,
    data_x,
    data_raw
) {
  
  if (
    inherits(
      model_obj,
      "catboost.Model"
    )
  ) {
    
    pool <- catboost.load_pool(
      data = data_x
    )
    
    prob <- catboost.predict(
      model_obj,
      pool,
      prediction_type = "Probability"
    )
    
    if (
      !is.null(dim(prob)) &&
      ncol(prob) >= 2
    ) {
      return(prob[, 2])
    } else {
      return(prob)
    }
    
  } else if (
    inherits(
      model_obj,
      "train"
    )
  ) {
    
    return(
      predict(
        model_obj,
        data_raw,
        type = "prob"
      )$Case
    )
    
  } else if (
    inherits(
      model_obj,
      "xgb.Booster"
    )
  ) {
    
    return(
      predict(
        model_obj,
        data_x
      )
    )
    
  } else if (
    inherits(
      model_obj,
      "lgb.Booster"
    )
  ) {
    
    return(
      predict(
        model_obj,
        data_x
      )
    )
    
  } else {
    
    stop(
      "Unrecognized model object."
    )
  }
}
# ==============================================================================
# 6. Generate predictions from all locked models
# ==============================================================================

all_probs <- list()


for (model_name in names(models)) {
  
  cat(
    "Generating predictions:",
    model_name,
    "\n"
  )
  
  all_probs[[model_name]] <- list(
    
    train_p = get_probs(
      models[[model_name]],
      train_x,
      train_set
    ),
    
    test_p = get_probs(
      models[[model_name]],
      test_x,
      test_set
    )
  )
}


saveRDS(
  all_probs,
  file.path(
    results_dir,
    "locked_all_models_predictions.rds"
  )
)
# ==============================================================================
# 7. Determine and lock Youden thresholds in the training set
# ==============================================================================

train_thresholds <- list()


for (model_name in names(all_probs)) {
  
  roc_train <- roc(
    response = train_y,
    predictor =
      all_probs[[model_name]]$train_p,
    levels = c(0, 1),
    direction = "<",
    quiet = TRUE
  )
  
  threshold_info <- coords(
    roc_train,
    x = "best",
    best.method = "youden",
    ret = "threshold",
    transpose = FALSE
  )
  
  train_thresholds[[model_name]] <-
    as.numeric(
      threshold_info$threshold[1]
    )
}


saveRDS(
  train_thresholds,
  file.path(
    results_dir,
    "locked_train_thresholds.rds"
  )
)
# ==============================================================================
# 8. PR-AUC helper
# ==============================================================================

calculate_pr_auc <- function(
    probs,
    labels
) {
  
  scores <- data.frame(
    probs = probs,
    labels = labels
  ) %>%
    arrange(
      desc(probs)
    )
  
  tps <- cumsum(
    scores$labels
  )
  
  fps <- cumsum(
    1 - scores$labels
  )
  
  recall <- tps /
    sum(scores$labels)
  
  precision <- tps /
    (tps + fps)
  
  recall <- c(
    0,
    recall
  )
  
  precision <- c(
    precision[1],
    precision
  )
  
  sum(
    diff(recall) *
      (
        precision[-1] +
          precision[-length(precision)]
      ) / 2,
    na.rm = TRUE
  )
}
# ==============================================================================
# 9. Model-evaluation function
# ==============================================================================

evaluate_model <- function(
    probs,
    labels,
    model_name,
    dataset_name,
    threshold
) {
  
  eps <- 1e-6
  
  probs_safe <- pmin(
    pmax(
      probs,
      eps
    ),
    1 - eps
  )
  
  
  # ROC / AUC
  roc_obj <- roc(
    response = labels,
    predictor = probs,
    levels = c(0, 1),
    direction = "<",
    quiet = TRUE
  )
  
  auc_value <- as.numeric(
    auc(roc_obj)
  )
  
  auc_ci <- ci.auc(
    roc_obj,
    method = "delong"
  )
  
  
  # PR-AUC
  pr_auc <- calculate_pr_auc(
    probs,
    labels
  )
  
  
  # Classification
  pred_class <- factor(
    ifelse(
      probs >= threshold,
      1,
      0
    ),
    levels = c(0, 1)
  )
  
  actual_class <- factor(
    labels,
    levels = c(0, 1)
  )
  
  cm <- confusionMatrix(
    pred_class,
    actual_class,
    positive = "1"
  )
  
  
  # Brier score
  brier <- mean(
    (labels - probs)^2
  )
  
  
  # Calibration intercept and slope
  logit_p <- qlogis(
    probs_safe
  )
  
  calibration_model <- glm(
    labels ~ logit_p,
    family = binomial()
  )
  
  calibration_intercept <-
    as.numeric(
      coef(calibration_model)[1]
    )
  
  calibration_slope <-
    as.numeric(
      coef(calibration_model)[2]
    )
  
  
  # Calibration-in-the-large
  citl_model <- glm(
    labels ~
      1 +
      offset(logit_p),
    family = binomial()
  )
  
  citl <-
    as.numeric(
      coef(citl_model)[1]
    )
  
  
  data.frame(
    Model = model_name,
    DataSet = dataset_name,
    Threshold = threshold,
    AUC = auc_value,
    AUC_CI_Lower =
      as.numeric(auc_ci[1]),
    AUC_CI_Upper =
      as.numeric(auc_ci[3]),
    PR_AUC = pr_auc,
    Brier_Score = brier,
    Accuracy =
      as.numeric(
        cm$overall["Accuracy"]
      ),
    Sensitivity =
      as.numeric(
        cm$byClass["Sensitivity"]
      ),
    Specificity =
      as.numeric(
        cm$byClass["Specificity"]
      ),
    PPV =
      as.numeric(
        cm$byClass["Pos Pred Value"]
      ),
    NPV =
      as.numeric(
        cm$byClass["Neg Pred Value"]
      ),
    F1_Score =
      as.numeric(
        cm$byClass["F1"]
      ),
    Calibration_Intercept =
      calibration_intercept,
    Calibration_Slope =
      calibration_slope,
    Calibration_in_the_Large =
      citl
  )
}
# ==============================================================================
# 10. Evaluate all six locked models
# ==============================================================================

all_model_metrics <- data.frame()


for (model_name in names(all_probs)) {
  
  cat(
    "Evaluating:",
    model_name,
    "\n"
  )
  
  threshold <-
    train_thresholds[[model_name]]
  
  
  training_result <- evaluate_model(
    probs =
      all_probs[[model_name]]$train_p,
    labels =
      train_y,
    model_name =
      model_name,
    dataset_name =
      "Training",
    threshold =
      threshold
  )
  
  
  testing_result <- evaluate_model(
    probs =
      all_probs[[model_name]]$test_p,
    labels =
      test_y,
    model_name =
      model_name,
    dataset_name =
      "Testing",
    threshold =
      threshold
  )
  
  
  all_model_metrics <- bind_rows(
    all_model_metrics,
    training_result,
    testing_result
  )
}


write.csv(
  all_model_metrics,
  file.path(
    results_dir,
    "Table4_all_model_performance.csv"
  ),
  row.names = FALSE
)
# ==============================================================================
# 11. Paired DeLong comparison: CatBoost vs Logistic Regression
# ==============================================================================

roc_lr_test <- roc(
  response = test_y,
  predictor =
    all_probs$LR$test_p,
  levels = c(0, 1),
  direction = "<",
  quiet = TRUE
)

roc_cat_test <- roc(
  response = test_y,
  predictor =
    all_probs$CAT$test_p,
  levels = c(0, 1),
  direction = "<",
  quiet = TRUE
)


delong_cat_vs_lr <- roc.test(
  roc_cat_test,
  roc_lr_test,
  method = "delong",
  paired = TRUE
)


delong_result <- data.frame(
  
  Comparison =
    "CatBoost vs Logistic Regression",
  
  Dataset =
    "Testing",
  
  AUC_CatBoost =
    as.numeric(
      auc(roc_cat_test)
    ),
  
  AUC_LR =
    as.numeric(
      auc(roc_lr_test)
    ),
  
  AUC_Difference =
    as.numeric(
      auc(roc_cat_test)
    ) -
    as.numeric(
      auc(roc_lr_test)
    ),
  
  Z =
    as.numeric(
      delong_cat_vs_lr$statistic
    ),
  
  P_value =
    as.numeric(
      delong_cat_vs_lr$p.value
    )
)


write.csv(
  delong_result,
  file.path(
    results_dir,
    "CatBoost_vs_LR_DeLong.csv"
  ),
  row.names = FALSE
)
# ==============================================================================
# 12. ROC curves for all six models
# ==============================================================================

pdf(
  file.path(
    results_dir,
    "Figure3_ROC_all_models.pdf"
  ),
  width = 12,
  height = 6
)

par(
  mfrow = c(1, 2)
)


# Training ROC
first <- TRUE

for (model_name in names(all_probs)) {
  
  roc_obj <- roc(
    train_y,
    all_probs[[model_name]]$train_p,
    quiet = TRUE
  )
  
  if (first) {
    
    plot(
      roc_obj,
      main =
        "A: Training Set ROC Curves"
    )
    
    first <- FALSE
    
  } else {
    
    plot(
      roc_obj,
      add = TRUE
    )
  }
}


legend(
  "bottomright",
  legend = names(all_probs),
  lty = 1,
  bty = "n"
)


# Testing ROC
first <- TRUE

for (model_name in names(all_probs)) {
  
  roc_obj <- roc(
    test_y,
    all_probs[[model_name]]$test_p,
    quiet = TRUE
  )
  
  if (first) {
    
    plot(
      roc_obj,
      main =
        "B: Testing Set ROC Curves"
    )
    
    first <- FALSE
    
  } else {
    
    plot(
      roc_obj,
      add = TRUE
    )
  }
}


legend(
  "bottomright",
  legend = names(all_probs),
  lty = 1,
  bty = "n"
)

dev.off()
# ==============================================================================
# 13. Hosmer-Lemeshow test for CatBoost
# ==============================================================================

hl_cat <- hoslem.test(
  test_y,
  all_probs$CAT$test_p,
  g = 10
)


catboost_calibration_summary <- data.frame(
  
  Metric = c(
    "Brier_Score",
    "Calibration_Intercept",
    "Calibration_Slope",
    "HL_test_P"
  ),
  
  Value = c(
    
    all_model_metrics %>%
      filter(
        Model == "CAT",
        DataSet == "Testing"
      ) %>%
      pull(Brier_Score),
    
    all_model_metrics %>%
      filter(
        Model == "CAT",
        DataSet == "Testing"
      ) %>%
      pull(Calibration_Intercept),
    
    all_model_metrics %>%
      filter(
        Model == "CAT",
        DataSet == "Testing"
      ) %>%
      pull(Calibration_Slope),
    
    hl_cat$p.value
  )
)


write.csv(
  catboost_calibration_summary,
  file.path(
    results_dir,
    "CatBoost_testing_calibration_summary.csv"
  ),
  row.names = FALSE
)
# ==============================================================================
# 14. CatBoost calibration curve
# ==============================================================================

pdf(
  file.path(
    results_dir,
    "Figure4_CatBoost_calibration.pdf"
  ),
  width = 7,
  height = 7
)


val_obj <- val.prob(
  all_probs$CAT$test_p,
  test_y,
  m = 50,
  cex = 0.8,
  logistic.cal = TRUE,
  statloc = FALSE,
  riskdist = "calibrated"
)


abline(
  0,
  1,
  lty = 2
)


dev.off()
# ==============================================================================
# 15. Decision Curve Analysis
# ==============================================================================

dca_df <- data.frame(
  outcome = test_y,
  CAT =
    all_probs$CAT$test_p,
  XGB =
    all_probs$XGB$test_p,
  LGBM =
    all_probs$LGBM$test_p,
  RF =
    all_probs$RF$test_p,
  LASSO =
    all_probs$LASSO$test_p,
  LR =
    all_probs$LR$test_p
)


dca_result <- dca(
  outcome ~
    CAT +
    XGB +
    LGBM +
    RF +
    LASSO +
    LR,
  data = dca_df,
  thresholds =
    seq(
      0,
      0.99,
      by = 0.01
    )
)


pdf(
  file.path(
    results_dir,
    "Figure5_DCA_all_models.pdf"
  ),
  width = 7.5,
  height = 7
)


print(
  plot(
    dca_result
  ) +
    ggplot2::labs(
      x =
        "Threshold Probability",
      y =
        "Net Benefit"
    ) +
    ggplot2::theme_classic()
)


dev.off()
# ==============================================================================
# 16. Reproducibility summary
# ==============================================================================

cat("\n")
cat("============================================================\n")
cat("Locked-model evaluation reproducibility summary\n")
cat("============================================================\n")


test_summary <- all_model_metrics %>%
  filter(
    DataSet == "Testing"
  ) %>%
  select(
    Model,
    AUC,
    Brier_Score,
    F1_Score
  )


print(
  test_summary
)


cat(
  "\nCatBoost vs LR paired DeLong P =",
  format(
    delong_result$P_value,
    digits = 5
  ),
  "\n"
)


cat(
  "CatBoost HL-test P =",
  format(
    hl_cat$p.value,
    digits = 5
  ),
  "\n"
)


cat(
  "CatBoost training-set Youden threshold =",
  train_thresholds$CAT,
  "\n"
)


cat("============================================================\n")