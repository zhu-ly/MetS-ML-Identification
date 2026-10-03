# ==============================================================================
# 04_probability_stratification_sensitivity.R
#
# Sensitivity analysis:
# Alternative probability cutoffs for exploratory CatBoost probability strata
#
# Objectives:
# 1. Use the locked final CatBoost model without retraining.
# 2. Compare three predefined probability-stratification schemes:
#
#      Alternative A: 0.10 / 0.30
#      Original:      0.15 / 0.33
#      Alternative B: 0.20 / 0.40
#
# 3. Apply exactly the same schemes to:
#      - independent testing set
#      - external validation cohort
#
# 4. Summarize:
#      - number and proportion of participants
#      - MetS events
#      - observed MetS prevalence
#      - exact binomial 95% confidence interval
#
# IMPORTANT:
# - These are exploratory descriptive probability strata.
# - They are NOT optimized clinical decision thresholds.
# - The CatBoost model is NOT retrained.
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
# 2. Define the three exploratory cutoff schemes
# ==============================================================================


cutoff_schemes <- data.frame(
  
  Scheme = c(
    "Alternative A",
    "Original",
    "Alternative B"
  ),
  
  Low_Cutoff = c(
    0.10,
    0.15,
    0.20
  ),
  
  High_Cutoff = c(
    0.30,
    0.33,
    0.40
  ),
  
  stringsAsFactors = FALSE
)


# ==============================================================================
# 3. Function for assigning probability strata
#
# Low:
#   probability < lower cutoff
#
# Intermediate:
#   lower cutoff <= probability <= upper cutoff
#
# High:
#   probability > upper cutoff
# ==============================================================================


assign_probability_stratum <- function(
    probability,
    low_cutoff,
    high_cutoff
) {
  
  stratum <- dplyr::case_when(
    
    probability < low_cutoff ~
      "Low",
    
    probability <= high_cutoff ~
      "Intermediate",
    
    probability > high_cutoff ~
      "High"
  )
  
  
  factor(
    stratum,
    levels = c(
      "Low",
      "Intermediate",
      "High"
    )
  )
}


# ==============================================================================
# 4. Function for summarizing one cutoff scheme
# ==============================================================================


summarize_probability_scheme <- function(
    probability,
    outcome,
    dataset_name,
    scheme_name,
    low_cutoff,
    high_cutoff
) {
  
  probability_stratum <-
    assign_probability_stratum(
      probability = probability,
      low_cutoff = low_cutoff,
      high_cutoff = high_cutoff
    )
  
  
  dat <- data.frame(
    Probability_Stratum =
      probability_stratum,
    Outcome =
      outcome
  )
  
  
  overall_prevalence <- mean(
    dat$Outcome
  )
  
  
  result_list <- lapply(
    levels(
      dat$Probability_Stratum
    ),
    function(g) {
      
      index <- dat$Probability_Stratum == g
      
      n_g <- sum(
        index
      )
      
      
      event_g <- sum(
        dat$Outcome[index] == 1
      )
      
      
      prevalence_g <- event_g / n_g
      
      
      exact_ci <- binom.test(
        event_g,
        n_g
      )$conf.int
      
      
      data.frame(
        
        Dataset =
          dataset_name,
        
        Scheme =
          scheme_name,
        
        Low_Cutoff =
          low_cutoff,
        
        High_Cutoff =
          high_cutoff,
        
        Probability_Stratum =
          g,
        
        N =
          n_g,
        
        Proportion_of_Cohort =
          n_g / nrow(dat),
        
        MetS_Events =
          event_g,
        
        MetS_Prevalence =
          prevalence_g,
        
        Prevalence_Lower95 =
          exact_ci[1],
        
        Prevalence_Upper95 =
          exact_ci[2],
        
        Overall_Prevalence =
          overall_prevalence,
        
        Prevalence_Ratio_to_Overall =
          prevalence_g /
          overall_prevalence,
        
        stringsAsFactors = FALSE
      )
    }
  )
  
  
  dplyr::bind_rows(
    result_list
  )
}


# ==============================================================================
# 5. Load independent testing set
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
# 6. Load external validation model data
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
# 7. Load locked CatBoost model
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
# 8. Generate testing-set probabilities
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
# 9. Generate external-validation probabilities
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
  length(external_prob) ==
    nrow(external_data),
  all(
    external_prob >= 0 &
      external_prob <= 1
  )
)


# ==============================================================================
# 10. Run all three schemes in the testing set
# ==============================================================================


test_results_list <- vector(
  "list",
  nrow(cutoff_schemes)
)


for (
  i in seq_len(
    nrow(cutoff_schemes)
  )
) {
  
  test_results_list[[i]] <-
    summarize_probability_scheme(
      
      probability =
        test_prob,
      
      outcome =
        test_y,
      
      dataset_name =
        "Testing set",
      
      scheme_name =
        cutoff_schemes$Scheme[i],
      
      low_cutoff =
        cutoff_schemes$Low_Cutoff[i],
      
      high_cutoff =
        cutoff_schemes$High_Cutoff[i]
    )
}


test_results <- dplyr::bind_rows(
  test_results_list
)


# ==============================================================================
# 11. Run identical schemes in the external validation cohort
# ==============================================================================


external_results_list <- vector(
  "list",
  nrow(cutoff_schemes)
)


for (
  i in seq_len(
    nrow(cutoff_schemes)
  )
) {
  
  external_results_list[[i]] <-
    summarize_probability_scheme(
      
      probability =
        external_prob,
      
      outcome =
        external_y,
      
      dataset_name =
        "External validation",
      
      scheme_name =
        cutoff_schemes$Scheme[i],
      
      low_cutoff =
        cutoff_schemes$Low_Cutoff[i],
      
      high_cutoff =
        cutoff_schemes$High_Cutoff[i]
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


stopifnot(
  nrow(combined_results) == 18
)


# ==============================================================================
# 13. Reproducibility checks against Supplementary Table S11
#
# Check exact N and event counts first.
# These uniquely determine the observed prevalence.
# ==============================================================================


expected_counts <- data.frame(
  
  Dataset = c(
    rep(
      "Testing set",
      9
    ),
    rep(
      "External validation",
      9
    )
  ),
  
  Scheme = rep(
    rep(
      c(
        "Alternative A",
        "Original",
        "Alternative B"
      ),
      each = 3
    ),
    2
  ),
  
  Probability_Stratum = rep(
    c(
      "Low",
      "Intermediate",
      "High"
    ),
    6
  ),
  
  N = c(
    # Testing set: Alternative A
    1201,
    893,
    866,
    
    # Testing set: Original
    1494,
    717,
    749,
    
    # Testing set: Alternative B
    1703,
    735,
    522,
    
    # External validation: Alternative A
    99,
    139,
    315,
    
    # External validation: Original
    124,
    139,
    290,
    
    # External validation: Alternative B
    161,
    154,
    238
  ),
  
  MetS_Events = c(
    # Testing set: Alternative A
    57,
    160,
    442,
    
    # Testing set: Original
    88,
    170,
    401,
    
    # Testing set: Alternative B
    129,
    215,
    315,
    
    # External validation: Alternative A
    4,
    28,
    181,
    
    # External validation: Original
    7,
    33,
    173,
    
    # External validation: Alternative B
    12,
    39,
    162
  ),
  
  stringsAsFactors = FALSE
)


check_table <- combined_results %>%
  select(
    Dataset,
    Scheme,
    Probability_Stratum,
    N,
    MetS_Events
  )


if (
  !identical(
    as.character(
      check_table$Dataset
    ),
    as.character(
      expected_counts$Dataset
    )
  )
) {
  
  stop(
    "Reproducibility check failed: Dataset order differs from Supplementary Table S11."
  )
}


if (
  !identical(
    as.character(
      check_table$Scheme
    ),
    as.character(
      expected_counts$Scheme
    )
  )
) {
  
  stop(
    "Reproducibility check failed: cutoff-scheme order differs from Supplementary Table S11."
  )
}


if (
  !identical(
    as.character(
      check_table$Probability_Stratum
    ),
    as.character(
      expected_counts$Probability_Stratum
    )
  )
) {
  
  stop(
    "Reproducibility check failed: probability-stratum order differs from Supplementary Table S11."
  )
}


if (
  !all(
    check_table$N ==
    expected_counts$N
  )
) {
  
  stop(
    "Reproducibility check failed for stratum sample sizes."
  )
}


if (
  !all(
    check_table$MetS_Events ==
    expected_counts$MetS_Events
  )
) {
  
  stop(
    "Reproducibility check failed for MetS event counts."
  )
}


# ==============================================================================
# 14. Check observed prevalence against Supplementary Table S11
# ==============================================================================


expected_prevalence_percent <- c(
  
  # Testing set: Alternative A
  4.7,
  17.9,
  51.0,
  
  # Testing set: Original
  5.9,
  23.7,
  53.5,
  
  # Testing set: Alternative B
  7.6,
  29.3,
  60.3,
  
  # External validation: Alternative A
  4.0,
  20.1,
  57.5,
  
  # External validation: Original
  5.6,
  23.7,
  59.7,
  
  # External validation: Alternative B
  7.5,
  25.3,
  68.1
)


observed_prevalence_percent <- round(
  100 *
    combined_results$MetS_Prevalence,
  1
)


if (
  !all(
    observed_prevalence_percent ==
    expected_prevalence_percent
  )
) {
  
  stop(
    "Reproducibility check failed for observed MetS prevalence."
  )
}


# ==============================================================================
# 15. Check exact 95% confidence intervals against Supplementary Table S11
# ==============================================================================


expected_ci_lower_percent <- c(
  
  # Testing set: Alternative A
  3.6,
  15.5,
  47.7,
  
  # Testing set: Original
  4.8,
  20.6,
  49.9,
  
  # Testing set: Alternative B
  6.4,
  26.0,
  56.0,
  
  # External validation: Alternative A
  1.1,
  13.8,
  51.8,
  
  # External validation: Original
  2.3,
  16.9,
  53.8,
  
  # External validation: Alternative B
  3.9,
  18.7,
  61.7
)


expected_ci_upper_percent <- c(
  
  # Testing set: Alternative A
  6.1,
  20.6,
  54.4,
  
  # Testing set: Original
  7.2,
  27.0,
  57.2,
  
  # Testing set: Alternative B
  8.9,
  32.7,
  64.6,
  
  # External validation: Alternative A
  10.0,
  27.8,
  63.0,
  
  # External validation: Original
  11.3,
  31.7,
  65.3,
  
  # External validation: Alternative B
  12.7,
  33.0,
  73.9
)


observed_ci_lower_percent <- round(
  100 *
    combined_results$Prevalence_Lower95,
  1
)


observed_ci_upper_percent <- round(
  100 *
    combined_results$Prevalence_Upper95,
  1
)


if (
  !all(
    observed_ci_lower_percent ==
    expected_ci_lower_percent
  )
) {
  
  stop(
    "Reproducibility check failed for lower 95% confidence limits."
  )
}


if (
  !all(
    observed_ci_upper_percent ==
    expected_ci_upper_percent
  )
) {
  
  stop(
    "Reproducibility check failed for upper 95% confidence limits."
  )
}


# ==============================================================================
# 16. Save aggregate results only
# ==============================================================================


write.csv(
  combined_results,
  file.path(
    results_dir,
    "CatBoost_Probability_Stratification_Sensitivity.csv"
  ),
  row.names = FALSE
)


formatted_results <- combined_results %>%
  mutate(
    
    Cutoff_Scheme = paste0(
      sprintf(
        "%.2f",
        Low_Cutoff
      ),
      "/",
      sprintf(
        "%.2f",
        High_Cutoff
      )
    ),
    
    `n (%)` = paste0(
      N,
      " (",
      sprintf(
        "%.1f",
        100 *
          Proportion_of_Cohort
      ),
      "%)"
    ),
    
    `Observed MetS prevalence, % (95% CI)` =
      paste0(
        sprintf(
          "%.1f",
          100 *
            MetS_Prevalence
        ),
        " (",
        sprintf(
          "%.1f",
          100 *
            Prevalence_Lower95
        ),
        "–",
        sprintf(
          "%.1f",
          100 *
            Prevalence_Upper95
        ),
        ")"
      )
  ) %>%
  select(
    Dataset,
    Scheme,
    Cutoff_Scheme,
    Probability_Stratum,
    `n (%)`,
    MetS_Events,
    `Observed MetS prevalence, % (95% CI)`
  )


write.csv(
  formatted_results,
  file.path(
    results_dir,
    "CatBoost_Probability_Stratification_Sensitivity_Formatted.csv"
  ),
  row.names = FALSE
)


# ==============================================================================
# 17. Console summary
# ==============================================================================


cat("\n")
cat("============================================================\n")
cat("CatBoost probability-stratification sensitivity analysis\n")
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


cat("\n")
cat("Cutoff schemes\n")
cat("--------------\n")


print(
  cutoff_schemes,
  row.names = FALSE
)


cat("\n")
cat("Probability-stratification results\n")
cat("----------------------------------\n")


print(
  as.data.frame(
    formatted_results
  ),
  row.names = FALSE
)


cat("\n")
cat("Original 0.15 / 0.33 scheme\n")
cat("---------------------------\n")


original_results <- formatted_results %>%
  filter(
    Scheme == "Original"
  )


print(
  as.data.frame(
    original_results
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
  "Cutoffs optimized in testing set: No\n"
)


cat(
  "Cutoffs optimized in external cohort: No\n"
)


cat(
  "Patient-level predictions written to disk: No\n"
)


cat(
  "Exact binomial 95% CIs used: Yes\n"
)


cat(
  "Supplementary Table S11 reproduced: Yes\n"
)


cat("============================================================\n")