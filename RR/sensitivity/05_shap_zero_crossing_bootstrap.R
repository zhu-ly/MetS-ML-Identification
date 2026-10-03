# ==============================================================================
# 05_shap_zero_crossing_bootstrap.R
#
# Sensitivity analysis:
# Bootstrap stability of model-specific SHAP zero-crossing estimates
#
# Objectives:
# 1. Use the locked final CatBoost model and the training set.
# 2. Calculate native CatBoost SHAP values for the seven final predictors.
# 3. For BMI, HbA1c, UA, and pulse, estimate the point at which the
#    LOESS-smoothed SHAP dependence curve crosses SHAP = 0 from
#    negative to positive.
# 4. Restrict LOESS fitting and zero-crossing estimation to the
#    2.5th-97.5th percentile range of each feature.
# 5. Assess stability using 1,000 nonparametric bootstrap resamples of
#    paired feature-SHAP values while keeping the fitted model fixed.
# 6. Calculate percentile-based 95% bootstrap confidence intervals.
#
# IMPORTANT:
# - The CatBoost model is NOT retrained during bootstrap.
# - These zero-crossings are model-specific descriptive reference points.
# - They are NOT clinical diagnostic or intervention thresholds.
# - Raw participant-level SHAP values are NOT written to disk.
# - Raw bootstrap replicate values are NOT written to disk.
# ==============================================================================


library(dplyr)
library(catboost)


source("RR/main/00_setup.R")


set.seed(SEED)


# ==============================================================================
# 1. Analysis settings
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


target_vars <- c(
  "BMI",
  "HbA1c",
  "UA",
  "pulse"
)


label_map <- c(
  "BMI"   = "BMI (kg/m2)",
  "HbA1c" = "HbA1c (%)",
  "UA"    = "UA (mg/dL)",
  "pulse" = "Pulse (beats/min)"
)


LOESS_SPAN <- 0.8

GRID_N <- 10000

BOOTSTRAP_REPLICATES <- 1000


# ==============================================================================
# 2. Load training set
# ==============================================================================


train_file <- file.path(
  processed_dir,
  "train_set.rds"
)


if (
  !file.exists(train_file)
) {
  
  stop(
    paste0(
      "Cannot find training set: ",
      train_file
    )
  )
}


train_set <- readRDS(
  train_file
)


required_vars <- c(
  final_vars,
  "met_diagnosis"
)


missing_vars <- setdiff(
  required_vars,
  names(train_set)
)


if (
  length(missing_vars) > 0
) {
  
  stop(
    paste(
      "Training set is missing:",
      paste(
        missing_vars,
        collapse = ", "
      )
    )
  )
}


stopifnot(
  nrow(train_set) == 6907,
  !anyNA(
    train_set[
      ,
      final_vars
    ]
  )
)


X_train <- train_set[
  ,
  final_vars,
  drop = FALSE
]


# ==============================================================================
# 3. Load locked final CatBoost model
# ==============================================================================


model_file <- file.path(
  model_dir,
  "locked",
  "model_cat.rds"
)


if (
  !file.exists(model_file)
) {
  
  stop(
    paste0(
      "Cannot find locked CatBoost model: ",
      model_file
    )
  )
}


model_cat <- readRDS(
  model_file
)


# ==============================================================================
# 4. Calculate native CatBoost SHAP values
#
# CatBoost ShapValues returns:
#   - one contribution column per feature
#   - one final expected-value / baseline column
#
# The final baseline column is therefore removed.
# ==============================================================================


cat(
  "\nCalculating native CatBoost SHAP values...\n"
)


pool_shap <- catboost.load_pool(
  data = X_train
)


shap_contribution <-
  catboost.get_feature_importance(
    model_cat,
    pool = pool_shap,
    type = "ShapValues"
  )


if (
  nrow(shap_contribution) !=
  nrow(X_train)
) {
  
  stop(
    "SHAP row count does not match the training-set row count."
  )
}


if (
  ncol(shap_contribution) !=
  length(final_vars) + 1
) {
  
  stop(
    paste0(
      "Unexpected number of SHAP columns. Expected ",
      length(final_vars) + 1,
      " but obtained ",
      ncol(shap_contribution),
      "."
    )
  )
}


shap_matrix <- shap_contribution[
  ,
  -ncol(shap_contribution),
  drop = FALSE
]


colnames(shap_matrix) <- final_vars


stopifnot(
  nrow(shap_matrix) == 6907,
  ncol(shap_matrix) == 7,
  all(
    is.finite(
      shap_matrix
    )
  )
)


# ==============================================================================
# 5. Negative-to-positive zero-crossing function
#
# Procedure:
# 1. Remove non-finite feature/SHAP pairs.
# 2. Determine the feature-specific 2.5th and 97.5th percentiles.
# 3. Restrict LOESS fitting to this range.
# 4. Fit LOESS using span = 0.8.
# 5. Predict the smoothed curve over a dense grid.
# 6. Identify locations where the fitted curve changes from <= 0 to > 0.
# 7. Estimate the first negative-to-positive crossing by linear interpolation.
#
# The number of negative-to-positive crossings is also recorded.
# ==============================================================================


get_zero_crossing <- function(
    x,
    shap,
    span = LOESS_SPAN,
    grid_n = GRID_N
) {
  
  dat <- data.frame(
    x = as.numeric(x),
    shap = as.numeric(shap)
  )
  
  
  dat <- dat[
    is.finite(dat$x) &
      is.finite(dat$shap),
    ,
    drop = FALSE
  ]
  
  
  if (
    nrow(dat) < 20
  ) {
    
    return(
      list(
        crossing = NA_real_,
        n_crossings = 0L,
        lower_limit = NA_real_,
        upper_limit = NA_real_
      )
    )
  }
  
  
  x_limits <- quantile(
    dat$x,
    probs = c(
      0.025,
      0.975
    ),
    na.rm = TRUE,
    names = FALSE
  )
  
  
  df_smooth <- dat[
    dat$x >= x_limits[1] &
      dat$x <= x_limits[2],
    ,
    drop = FALSE
  ]
  
  
  if (
    nrow(df_smooth) < 20 ||
    length(
      unique(
        df_smooth$x
      )
    ) < 5
  ) {
    
    return(
      list(
        crossing = NA_real_,
        n_crossings = 0L,
        lower_limit = x_limits[1],
        upper_limit = x_limits[2]
      )
    )
  }
  
  
  loess_fit <- tryCatch(
    
    loess(
      shap ~ x,
      data = df_smooth,
      span = span
    ),
    
    error = function(e) {
      NULL
    }
  )
  
  
  if (
    is.null(loess_fit)
  ) {
    
    return(
      list(
        crossing = NA_real_,
        n_crossings = 0L,
        lower_limit = x_limits[1],
        upper_limit = x_limits[2]
      )
    )
  }
  
  
  grid_x <- seq(
    min(
      df_smooth$x
    ),
    max(
      df_smooth$x
    ),
    length.out = grid_n
  )
  
  
  pred_y <- suppressWarnings(
    predict(
      loess_fit,
      newdata = data.frame(
        x = grid_x
      )
    )
  )
  
  
  valid <- is.finite(
    pred_y
  )
  
  
  grid_x <- grid_x[
    valid
  ]
  
  
  pred_y <- pred_y[
    valid
  ]
  
  
  if (
    length(pred_y) < 2
  ) {
    
    return(
      list(
        crossing = NA_real_,
        n_crossings = 0L,
        lower_limit = x_limits[1],
        upper_limit = x_limits[2]
      )
    )
  }
  
  
  crossing_idx <- which(
    pred_y[
      -length(pred_y)
    ] <= 0 &
      pred_y[-1] > 0
  )
  
  
  if (
    length(crossing_idx) == 0
  ) {
    
    return(
      list(
        crossing = NA_real_,
        n_crossings = 0L,
        lower_limit = x_limits[1],
        upper_limit = x_limits[2]
      )
    )
  }
  
  
  n_crossings <- length(
    crossing_idx
  )
  
  
  i <- crossing_idx[1]
  
  
  x1 <- grid_x[i]
  x2 <- grid_x[i + 1]
  
  y1 <- pred_y[i]
  y2 <- pred_y[i + 1]
  
  
  zero_x <- x1 +
    (0 - y1) *
    (x2 - x1) /
    (y2 - y1)
  
  
  list(
    crossing = as.numeric(
      zero_x
    ),
    n_crossings = as.integer(
      n_crossings
    ),
    lower_limit = x_limits[1],
    upper_limit = x_limits[2]
  )
}


# ==============================================================================
# 6. Calculate zero-crossing estimates in the original training sample
# ==============================================================================


cat(
  "Calculating original training-set zero-crossings...\n"
)


original_list <- lapply(
  target_vars,
  function(v) {
    
    z <- get_zero_crossing(
      x = X_train[[v]],
      shap = shap_matrix[, v],
      span = LOESS_SPAN,
      grid_n = GRID_N
    )
    
    
    data.frame(
      Predictor = v,
      Predictor_Label =
        unname(
          label_map[v]
        ),
      SHAP_Zero_Crossing =
        z$crossing,
      Number_of_Negative_to_Positive_Crossings =
        z$n_crossings,
      Feature_2.5th_Percentile =
        z$lower_limit,
      Feature_97.5th_Percentile =
        z$upper_limit,
      stringsAsFactors = FALSE
    )
  }
)


original_results <- dplyr::bind_rows(
  original_list
)


if (
  any(
    !is.finite(
      original_results$SHAP_Zero_Crossing
    )
  )
) {
  
  stop(
    "At least one original SHAP zero-crossing could not be estimated."
  )
}


# ==============================================================================
# 7. Check original zero-crossing estimates against Supplementary Table S6
# ==============================================================================


expected_original <- c(
  BMI = 23.5388,
  HbA1c = 6.2464,
  UA = 4.9728,
  pulse = 76.0594
)


observed_original <- setNames(
  original_results$SHAP_Zero_Crossing,
  original_results$Predictor
)


for (
  v in target_vars
) {
  
  if (
    abs(
      observed_original[[v]] -
      expected_original[[v]]
    ) > 0.0001
  ) {
    
    stop(
      paste0(
        "Original zero-crossing check failed for ",
        v,
        ". Observed = ",
        sprintf(
          "%.6f",
          observed_original[[v]]
        ),
        "; expected approximately = ",
        sprintf(
          "%.4f",
          expected_original[[v]]
        ),
        "."
      )
    )
  }
}


# ==============================================================================
# 8. Bootstrap function
#
# The fitted CatBoost model and calculated SHAP values remain fixed.
# Each bootstrap replicate resamples paired feature-SHAP observations
# with replacement.
# ==============================================================================


bootstrap_zero_crossing <- function(
    x,
    shap,
    B = BOOTSTRAP_REPLICATES,
    span = LOESS_SPAN,
    grid_n = GRID_N
) {
  
  n <- length(
    x
  )
  
  
  boot_estimates <- rep(
    NA_real_,
    B
  )
  
  
  boot_n_crossings <- rep(
    NA_integer_,
    B
  )
  
  
  for (
    b in seq_len(B)
  ) {
    
    idx <- sample(
      seq_len(n),
      size = n,
      replace = TRUE
    )
    
    
    z <- get_zero_crossing(
      x = x[idx],
      shap = shap[idx],
      span = span,
      grid_n = grid_n
    )
    
    
    boot_estimates[b] <-
      z$crossing
    
    
    boot_n_crossings[b] <-
      z$n_crossings
    
    
    if (
      b %% 100 == 0
    ) {
      
      cat(
        "  Bootstrap replicate:",
        b,
        "/",
        B,
        "\n"
      )
    }
  }
  
  
  successful <- boot_estimates[
    is.finite(
      boot_estimates
    )
  ]
  
  
  success_n <- length(
    successful
  )
  
  
  success_rate <- success_n / B
  
  
  if (
    success_n == 0
  ) {
    
    return(
      list(
        estimates =
          boot_estimates,
        n_crossings =
          boot_n_crossings,
        median =
          NA_real_,
        lower =
          NA_real_,
        upper =
          NA_real_,
        success_n =
          0L,
        success_rate =
          0,
        multiple_crossing_n =
          0L,
        multiple_crossing_rate =
          0
      )
    )
  }
  
  
  ci_values <- quantile(
    successful,
    probs = c(
      0.025,
      0.50,
      0.975
    ),
    na.rm = TRUE,
    names = FALSE
  )
  
  
  multiple_crossing_n <- sum(
    boot_n_crossings > 1,
    na.rm = TRUE
  )
  
  
  multiple_crossing_rate <-
    multiple_crossing_n / B
  
  
  list(
    estimates =
      boot_estimates,
    n_crossings =
      boot_n_crossings,
    median =
      ci_values[2],
    lower =
      ci_values[1],
    upper =
      ci_values[3],
    success_n =
      success_n,
    success_rate =
      success_rate,
    multiple_crossing_n =
      multiple_crossing_n,
    multiple_crossing_rate =
      multiple_crossing_rate
  )
}


# ==============================================================================
# 9. Run 1,000 bootstrap replicates for each target predictor
#
# IMPORTANT:
# Resetting the seed here reproduces the original 09A bootstrap sequence.
# ==============================================================================


set.seed(SEED)


bootstrap_summary_list <- list()


for (
  v in target_vars
) {
  
  cat("\n")
  cat("============================================================\n")
  cat(
    "Bootstrap predictor:",
    v,
    "\n"
  )
  cat(
    "Replicates:",
    BOOTSTRAP_REPLICATES,
    "\n"
  )
  cat("============================================================\n")
  
  
  res <- bootstrap_zero_crossing(
    x = X_train[[v]],
    shap = shap_matrix[, v],
    B = BOOTSTRAP_REPLICATES,
    span = LOESS_SPAN,
    grid_n = GRID_N
  )
  
  
  original_estimate <-
    original_results$SHAP_Zero_Crossing[
      original_results$Predictor == v
    ]
  
  
  original_n_crossings <-
    original_results$Number_of_Negative_to_Positive_Crossings[
      original_results$Predictor == v
    ]
  
  
  bootstrap_summary_list[[v]] <-
    data.frame(
      
      Predictor =
        v,
      
      Predictor_Label =
        unname(
          label_map[v]
        ),
      
      SHAP_Zero_Crossing =
        original_estimate,
      
      Original_Number_of_Crossings =
        original_n_crossings,
      
      Bootstrap_Median =
        res$median,
      
      Bootstrap_CI_Lower_2.5 =
        res$lower,
      
      Bootstrap_CI_Upper_97.5 =
        res$upper,
      
      Successful_Bootstrap_Replicates =
        res$success_n,
      
      Total_Bootstrap_Replicates =
        BOOTSTRAP_REPLICATES,
      
      Bootstrap_Success_Rate =
        res$success_rate,
      
      Multiple_Crossing_Replicates =
        res$multiple_crossing_n,
      
      Multiple_Crossing_Rate =
        res$multiple_crossing_rate,
      
      stringsAsFactors = FALSE
    )
}


bootstrap_table <- dplyr::bind_rows(
  bootstrap_summary_list
)


# ==============================================================================
# 10. Reproducibility checks against current Supplementary Table S6
# ==============================================================================


expected_ci_lower <- c(
  BMI = 23.5310,
  HbA1c = 6.2446,
  UA = 4.9653,
  pulse = 75.9220
)


expected_ci_upper <- c(
  BMI = 23.5480,
  HbA1c = 6.2523,
  UA = 4.9827,
  pulse = 76.1179
)


for (
  v in target_vars
) {
  
  row_v <- bootstrap_table[
    bootstrap_table$Predictor == v,
    ,
    drop = FALSE
  ]
  
  
  if (
    nrow(row_v) != 1
  ) {
    
    stop(
      paste0(
        "Unexpected bootstrap summary structure for ",
        v,
        "."
      )
    )
  }
  
  
  if (
    row_v$Successful_Bootstrap_Replicates !=
    BOOTSTRAP_REPLICATES
  ) {
    
    stop(
      paste0(
        "Not all bootstrap replicates were successful for ",
        v,
        ". Successful = ",
        row_v$Successful_Bootstrap_Replicates,
        "/",
        BOOTSTRAP_REPLICATES,
        "."
      )
    )
  }
  
  
  if (
    round(
      row_v$Bootstrap_CI_Lower_2.5,
      4
    ) !=
    expected_ci_lower[[v]]
  ) {
    
    stop(
      paste0(
        "Bootstrap lower CI check failed for ",
        v,
        ". Observed = ",
        sprintf(
          "%.4f",
          row_v$Bootstrap_CI_Lower_2.5
        ),
        "; expected = ",
        sprintf(
          "%.4f",
          expected_ci_lower[[v]]
        ),
        "."
      )
    )
  }
  
  
  if (
    round(
      row_v$Bootstrap_CI_Upper_97.5,
      4
    ) !=
    expected_ci_upper[[v]]
  ) {
    
    stop(
      paste0(
        "Bootstrap upper CI check failed for ",
        v,
        ". Observed = ",
        sprintf(
          "%.4f",
          row_v$Bootstrap_CI_Upper_97.5
        ),
        "; expected = ",
        sprintf(
          "%.4f",
          expected_ci_upper[[v]]
        ),
        "."
      )
    )
  }
}


# Current Supplementary Table S6 states that all bootstrap
# replicates yielded a single negative-to-positive zero-crossing.

if (
  any(
    bootstrap_table$Multiple_Crossing_Replicates != 0
  )
) {
  
  stop(
    paste0(
      "At least one bootstrap replicate produced multiple ",
      "negative-to-positive zero-crossings, which does not match ",
      "the current Supplementary Table S6 legend."
    )
  )
}


if (
  any(
    original_results$Number_of_Negative_to_Positive_Crossings != 1
  )
) {
  
  stop(
    paste0(
      "At least one original dependence curve did not yield exactly ",
      "one negative-to-positive zero-crossing."
    )
  )
}


# ==============================================================================
# 11. Save aggregate reproducibility outputs only
# ==============================================================================


write.csv(
  original_results,
  file.path(
    results_dir,
    "SHAP_ZeroCrossing_Original_Estimates.csv"
  ),
  row.names = FALSE
)


write.csv(
  bootstrap_table,
  file.path(
    results_dir,
    "SHAP_ZeroCrossing_Bootstrap_Summary.csv"
  ),
  row.names = FALSE
)


supplementary_s6 <- bootstrap_table %>%
  transmute(
    
    Predictor =
      Predictor_Label,
    
    `SHAP zero-crossing estimate` =
      sprintf(
        "%.4f",
        SHAP_Zero_Crossing
      ),
    
    `Bootstrap 95% CI` =
      paste0(
        sprintf(
          "%.4f",
          Bootstrap_CI_Lower_2.5
        ),
        "–",
        sprintf(
          "%.4f",
          Bootstrap_CI_Upper_97.5
        )
      ),
    
    `Successful bootstrap replicates` =
      paste0(
        Successful_Bootstrap_Replicates,
        "/",
        Total_Bootstrap_Replicates
      )
  )


write.csv(
  supplementary_s6,
  file.path(
    results_dir,
    "Supplementary_Table_S6_SHAP_ZeroCrossing_Bootstrap.csv"
  ),
  row.names = FALSE
)


# ==============================================================================
# 12. Console summary
# ==============================================================================


cat("\n\n")
cat("============================================================\n")
cat("SHAP zero-crossing bootstrap sensitivity analysis\n")
cat("============================================================\n")


cat(
  "Training N =",
  nrow(X_train),
  "\n"
)


cat(
  "Final predictors =",
  paste(
    final_vars,
    collapse = ", "
  ),
  "\n"
)


cat(
  "Target predictors =",
  paste(
    target_vars,
    collapse = ", "
  ),
  "\n"
)


cat(
  "LOESS span =",
  LOESS_SPAN,
  "\n"
)


cat(
  "Feature fitting range = 2.5th-97.5th percentile\n"
)


cat(
  "Bootstrap replicates per predictor =",
  BOOTSTRAP_REPLICATES,
  "\n"
)


cat(
  "CatBoost model retrained during bootstrap: No\n"
)


cat("\n")
cat("Supplementary Table S6\n")
cat("----------------------\n")


print(
  as.data.frame(
    supplementary_s6
  ),
  row.names = FALSE
)


cat("\n")
cat("Additional stability checks\n")
cat("---------------------------\n")


stability_table <- bootstrap_table %>%
  transmute(
    
    Predictor,
    
    Bootstrap_Median =
      sprintf(
        "%.4f",
        Bootstrap_Median
      ),
    
    Successful_Replicates =
      paste0(
        Successful_Bootstrap_Replicates,
        "/",
        Total_Bootstrap_Replicates
      ),
    
    Original_Crossings =
      Original_Number_of_Crossings,
    
    Multiple_Crossing_Replicates =
      Multiple_Crossing_Replicates
  )


print(
  as.data.frame(
    stability_table
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
  "Native CatBoost SHAP values used: Yes\n"
)


cat(
  "Model retrained during bootstrap: No\n"
)


cat(
  "Paired feature-SHAP bootstrap used: Yes\n"
)


cat(
  "Percentile-based 95% bootstrap CI used: Yes\n"
)


cat(
  "Raw participant-level SHAP values written to disk: No\n"
)


cat(
  "Raw bootstrap replicate values written to disk: No\n"
)


cat(
  "Supplementary Table S6 reproduced: Yes\n"
)


cat("============================================================\n")