# ==============================================================================
# 05_train_models.R
# Train and tune the six candidate models
# ==============================================================================

library(caret)
library(glmnet)
library(randomForest)
library(xgboost)
library(lightgbm)
library(catboost)
library(dplyr)

source("RR/main/00_setup.R")


# ==============================================================================
# 1. Paths and training data
# ==============================================================================

train_file <- file.path(
  processed_dir,
  "train_set.rds"
)

if (!file.exists(train_file)) {
  stop(
    "Training dataset was not found. ",
    "Run RR/main/03_descriptive_analysis_and_split.R first."
  )
}

train_set <- readRDS(train_file)

retrained_model_dir <- file.path(
  model_dir,
  "retrained"
)

dir.create(
  retrained_model_dir,
  showWarnings = FALSE,
  recursive = TRUE
)


# Verify training cohort
stopifnot(
  nrow(train_set) == 6907,
  sum(train_set$met_diagnosis == "1") == 1538,
  all(complete.cases(train_set))
)
# ==============================================================================
# 2. Final predictor set
# ==============================================================================

stopifnot(
  identical(
    final_predictors,
    c(
      "age",
      "BMI",
      "pulse",
      "BUN",
      "TC",
      "UA",
      "HbA1c"
    )
  )
)

train_x_mat <- as.matrix(
  train_set[, final_predictors]
)


# Convert outcome for model training
train_set$met_diagnosis <- factor(
  ifelse(
    train_set$met_diagnosis == 1,
    "Case",
    "Control"
  ),
  levels = c(
    "Control",
    "Case"
  )
)

train_y_num <- ifelse(
  train_set$met_diagnosis == "Case",
  1,
  0
)


formula_model <- as.formula(
  paste(
    "met_diagnosis ~",
    paste(
      final_predictors,
      collapse = " + "
    )
  )
)
# ==============================================================================
# 3. Cross-validation setup for caret-based models
# ==============================================================================

set.seed(SEED)

folds_idx <- createFolds(
  train_set$met_diagnosis,
  k = 10,
  list = TRUE,
  returnTrain = FALSE
)

train_control <- trainControl(
  method = "cv",
  indexOut = folds_idx,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final"
)
# ==============================================================================
# 4. Logistic Regression
# ==============================================================================

cat("Training 1/6: Logistic Regression...\n")

model_lr <- train(
  formula_model,
  data = train_set,
  method = "glm",
  family = "binomial",
  trControl = train_control,
  metric = "ROC"
)
# ==============================================================================
# 5. LASSO
# ==============================================================================

cat("Training 2/6: LASSO...\n")

cv_lasso_model <- cv.glmnet(
  x = train_x_mat,
  y = train_y_num,
  family = "binomial",
  alpha = 1,
  nfolds = 10
)

model_lasso <- train(
  formula_model,
  data = train_set,
  method = "glmnet",
  tuneGrid = expand.grid(
    alpha = 1,
    lambda = cv_lasso_model$lambda.1se
  ),
  trControl = train_control,
  metric = "ROC"
)
# ==============================================================================
# 6. Random Forest
# ==============================================================================

cat("Training 3/6: Random Forest...\n")

rf_grid <- expand.grid(
  mtry = c(
    2,
    3,
    4,
    5
  )
)

model_rf <- train(
  formula_model,
  data = train_set,
  method = "rf",
  ntree = 500,
  nodesize = 50,
  tuneGrid = rf_grid,
  trControl = train_control,
  metric = "ROC"
)
# ==============================================================================
# 7. XGBoost
# ==============================================================================

cat("Training 4/6: XGBoost...\n")

dtrain_xgb <- xgb.DMatrix(
  data = train_x_mat,
  label = train_y_num
)


# Initial CV to identify an iteration benchmark
xgb_cv_base <- xgb.cv(
  params = list(
    eta = 0.1,
    max_depth = 6,
    objective = "binary:logistic",
    eval_metric = "auc",
    seed = SEED
  ),
  data = dtrain_xgb,
  nrounds = 1000,
  nfold = 10,
  early_stopping_rounds = 20,
  verbose = 0
)


best_n_xgb <- if (
  !is.null(xgb_cv_base$best_iteration)
) {
  xgb_cv_base$best_iteration
} else {
  which.max(
    xgb_cv_base$evaluation_log$test_auc_mean
  )
}


xgb_grid <- expand.grid(
  max_depth = c(3, 6),
  subsample = c(0.8, 1.0)
)


xgb_results <- data.frame()


for (i in seq_len(nrow(xgb_grid))) {
  
  n_limit <- max(
    10,
    round(best_n_xgb * 1.5)
  )
  
  cv <- xgb.cv(
    params = list(
      eta = 0.1,
      max_depth = xgb_grid$max_depth[i],
      subsample = xgb_grid$subsample[i],
      objective = "binary:logistic",
      eval_metric = "auc",
      seed = SEED
    ),
    data = dtrain_xgb,
    nrounds = n_limit,
    nfold = 10,
    early_stopping_rounds = 20,
    verbose = 0
  )
  
  best_iter <- if (
    !is.null(cv$best_iteration)
  ) {
    cv$best_iteration
  } else {
    which.max(
      cv$evaluation_log$test_auc_mean
    )
  }
  
  xgb_results <- rbind(
    xgb_results,
    data.frame(
      max_depth = xgb_grid$max_depth[i],
      subsample = xgb_grid$subsample[i],
      AUC = max(
        cv$evaluation_log$test_auc_mean
      ),
      CV_best_iteration = best_iter
    )
  )
}
best_xgb_row <- xgb_results[
  which.max(xgb_results$AUC),
]

xgb_final_nrounds <- max(
  1,
  round(
    best_xgb_row$CV_best_iteration *
      1.10
  )
)


model_xgb <- xgb.train(
  params = list(
    eta = 0.1,
    max_depth = best_xgb_row$max_depth,
    subsample = best_xgb_row$subsample,
    objective = "binary:logistic"
  ),
  data = dtrain_xgb,
  nrounds = xgb_final_nrounds
)
# ==============================================================================
# 8. LightGBM
# ==============================================================================

cat("Training 5/6: LightGBM...\n")

dtrain_lgb <- lgb.Dataset(
  data = train_x_mat,
  label = train_y_num
)


lgb_cv_base <- lgb.cv(
  params = list(
    objective = "binary",
    learning_rate = 0.1,
    num_leaves = 31,
    seed = SEED,
    metric = "auc",
    verbose = -1
  ),
  data = dtrain_lgb,
  nrounds = 1000,
  nfold = 10,
  early_stopping_rounds = 20
)


lgb_grid <- expand.grid(
  num_leaves = c(15, 31),
  feature_fraction = c(0.8, 1.0),
  bagging_fraction = c(0.8, 1.0)
)


lgb_results <- data.frame()


for (i in seq_len(nrow(lgb_grid))) {
  
  cv <- lgb.cv(
    params = list(
      objective = "binary",
      learning_rate = 0.1,
      seed = SEED,
      metric = "auc",
      num_leaves =
        lgb_grid$num_leaves[i],
      feature_fraction =
        lgb_grid$feature_fraction[i],
      bagging_fraction =
        lgb_grid$bagging_fraction[i],
      bagging_freq = 5,
      verbose = -1
    ),
    data = dtrain_lgb,
    nrounds = max(
      10,
      round(
        lgb_cv_base$best_iter *
          1.5
      )
    ),
    nfold = 10,
    early_stopping_rounds = 20
  )
  
  lgb_results <- rbind(
    lgb_results,
    data.frame(
      num_leaves =
        lgb_grid$num_leaves[i],
      feature_fraction =
        lgb_grid$feature_fraction[i],
      bagging_fraction =
        lgb_grid$bagging_fraction[i],
      AUC = max(
        unlist(
          cv$record_evals$valid$auc$eval
        )
      ),
      CV_best_iteration =
        cv$best_iter
    )
  )
}
best_lgb_row <- lgb_results[
  which.max(lgb_results$AUC),
]

lgb_final_nrounds <- max(
  1,
  round(
    best_lgb_row$CV_best_iteration *
      1.10
  )
)


model_lgbm <- lgb.train(
  params = list(
    objective = "binary",
    learning_rate = 0.1,
    seed = SEED,
    num_leaves =
      best_lgb_row$num_leaves,
    feature_fraction =
      best_lgb_row$feature_fraction,
    bagging_fraction =
      best_lgb_row$bagging_fraction,
    bagging_freq = 5
  ),
  data = dtrain_lgb,
  nrounds = lgb_final_nrounds
)
# ==============================================================================
# 9. CatBoost
# ==============================================================================

cat("Training 6/6: CatBoost...\n")

train_pool <- catboost.load_pool(
  data = train_x_mat,
  label = train_y_num
)


cat_cv_base <- catboost.cv(
  train_pool,
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


cat_grid <- expand.grid(
  depth = c(4, 6),
  l2_leaf_reg = c(1, 3, 5)
)


cat_results <- data.frame()


for (i in seq_len(nrow(cat_grid))) {
  
  cv <- catboost.cv(
    train_pool,
    params = list(
      iterations = max(
        10,
        round(
          which.max(
            cat_cv_base$test.AUC.mean
          ) * 1.5
        )
      ),
      learning_rate = 0.1,
      depth = cat_grid$depth[i],
      l2_leaf_reg = cat_grid$l2_leaf_reg[i],
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
  
  cat_results <- rbind(
    cat_results,
    data.frame(
      depth =
        cat_grid$depth[i],
      l2_leaf_reg =
        cat_grid$l2_leaf_reg[i],
      AUC =
        max(cv$test.AUC.mean),
      CV_best_iteration =
        which.max(
          cv$test.AUC.mean
        )
    )
  )
}
best_cat_row <- cat_results[
  which.max(cat_results$AUC),
]

cat_final_iterations <- max(
  1,
  round(
    best_cat_row$CV_best_iteration *
      1.10
  )
)


model_cat_retrained <- catboost.train(
  train_pool,
  params = list(
    iterations = cat_final_iterations,
    learning_rate = 0.1,
    depth = best_cat_row$depth,
    l2_leaf_reg = best_cat_row$l2_leaf_reg,
    loss_function = "Logloss",
    random_seed = SEED,
    border_count = 254,
    logging_level = "Silent",
    allow_writing_files = FALSE
  )
)
# ==============================================================================
# 10. Save retrained model objects
# ==============================================================================

saveRDS(
  model_lr,
  file.path(
    retrained_model_dir,
    "model_lr.rds"
  )
)

saveRDS(
  model_lasso,
  file.path(
    retrained_model_dir,
    "model_lasso.rds"
  )
)

saveRDS(
  model_rf,
  file.path(
    retrained_model_dir,
    "model_rf.rds"
  )
)

saveRDS(
  model_xgb,
  file.path(
    retrained_model_dir,
    "model_xgb.rds"
  )
)

saveRDS(
  model_lgbm,
  file.path(
    retrained_model_dir,
    "model_lgbm.rds"
  )
)

saveRDS(
  model_cat_retrained,
  file.path(
    retrained_model_dir,
    "model_cat_retrained.rds"
  )
)
# ==============================================================================
# 11. Save tuning results
# ==============================================================================

write.csv(
  xgb_results,
  file.path(
    results_dir,
    "XGBoost_tuning_results.csv"
  ),
  row.names = FALSE
)

write.csv(
  lgb_results,
  file.path(
    results_dir,
    "LightGBM_tuning_results.csv"
  ),
  row.names = FALSE
)

write.csv(
  cat_results,
  file.path(
    results_dir,
    "CatBoost_tuning_results.csv"
  ),
  row.names = FALSE
)
# ==============================================================================
# 12. Logistic-regression coefficient table
# ==============================================================================

raw_lr_model <- model_lr$finalModel

lr_summary <- summary(
  raw_lr_model
)


table3 <- data.frame(
  Predictor =
    rownames(
      lr_summary$coefficients
    ),
  Beta =
    lr_summary$coefficients[, 1],
  SE =
    lr_summary$coefficients[, 2],
  P_value =
    lr_summary$coefficients[, 4]
) %>%
  filter(
    Predictor != "(Intercept)"
  ) %>%
  mutate(
    OR = exp(Beta),
    CI_lower =
      exp(Beta - 1.96 * SE),
    CI_upper =
      exp(Beta + 1.96 * SE)
  )


write.csv(
  table3,
  file.path(
    results_dir,
    "Table3_logistic_regression.csv"
  ),
  row.names = FALSE
)
# ==============================================================================
# 13. Save selected hyperparameter summary
# ==============================================================================

selected_hyperparameters <- data.frame(
  Model = c(
    "Random Forest",
    "XGBoost",
    "LightGBM",
    "CatBoost"
  ),
  
  Selected_parameters = c(
    
    paste0(
      "mtry=",
      model_rf$bestTune$mtry,
      "; ntree=500; nodesize=50"
    ),
    
    paste0(
      "eta=0.1; max_depth=",
      best_xgb_row$max_depth,
      "; subsample=",
      best_xgb_row$subsample,
      "; nrounds=",
      xgb_final_nrounds
    ),
    
    paste0(
      "learning_rate=0.1; num_leaves=",
      best_lgb_row$num_leaves,
      "; feature_fraction=",
      best_lgb_row$feature_fraction,
      "; bagging_fraction=",
      best_lgb_row$bagging_fraction,
      "; bagging_freq=5; nrounds=",
      lgb_final_nrounds
    ),
    
    paste0(
      "learning_rate=0.1; depth=",
      best_cat_row$depth,
      "; l2_leaf_reg=",
      best_cat_row$l2_leaf_reg,
      "; border_count=254; iterations=",
      cat_final_iterations
    )
  )
)


write.csv(
  selected_hyperparameters,
  file.path(
    results_dir,
    "selected_hyperparameters_retrained.csv"
  ),
  row.names = FALSE
)
# ==============================================================================
# 14. Reproducibility summary
# ==============================================================================

cat("\n")
cat("============================================================\n")
cat("Model-training reproducibility summary\n")
cat("============================================================\n")

cat(
  "Random Forest mtry =",
  model_rf$bestTune$mtry,
  "\n"
)

cat(
  "XGBoost: max_depth =",
  best_xgb_row$max_depth,
  "; subsample =",
  best_xgb_row$subsample,
  "; nrounds =",
  xgb_final_nrounds,
  "\n"
)

cat(
  "LightGBM: num_leaves =",
  best_lgb_row$num_leaves,
  "; feature_fraction =",
  best_lgb_row$feature_fraction,
  "; bagging_fraction =",
  best_lgb_row$bagging_fraction,
  "; nrounds =",
  lgb_final_nrounds,
  "\n"
)

cat(
  "CatBoost: depth =",
  best_cat_row$depth,
  "; l2_leaf_reg =",
  best_cat_row$l2_leaf_reg,
  "; iterations =",
  cat_final_iterations,
  "\n"
)

cat("============================================================\n")
# ==============================================================================
# 15. Save hyperparameters of the locked models reported in the manuscript
# ==============================================================================

locked_hyperparameters <- data.frame(
  Model = c(
    "Random Forest",
    "XGBoost",
    "LightGBM",
    "CatBoost"
  ),
  
  Reported_locked_parameters = c(
    
    "mtry=5; ntree=500; nodesize=50",
    
    paste0(
      "eta=0.1; max_depth=3; subsample=0.8; ",
      "nrounds=668; objective=binary:logistic"
    ),
    
    paste0(
      "learning_rate=0.1; num_leaves=15; ",
      "feature_fraction=1.0; bagging_fraction=0.8; ",
      "bagging_freq=5; nrounds=31; objective=binary"
    ),
    
    paste0(
      "learning_rate=0.1; depth=4; l2_leaf_reg=5; ",
      "border_count=254; iterations=73; loss_function=Logloss"
    )
  ),
  
  Source = c(
    "Locked final model / Supplementary Table S1",
    "Locked final model metadata and tree-count recovery / Supplementary Table S1",
    "Locked final model / Supplementary Table S1",
    "Locked final model / Supplementary Table S1"
  )
)

write.csv(
  locked_hyperparameters,
  file.path(
    results_dir,
    "locked_model_hyperparameters.csv"
  ),
  row.names = FALSE
)