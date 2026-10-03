# ==============================================================================
# 03_descriptive_analysis_and_split.R
# Descriptive analysis and stratified training/testing split
# ==============================================================================

library(dplyr)
library(tidyr)
library(caret)

source("RR/main/00_setup.R")


# ==============================================================================
# 1. Load final complete-case analytic dataset
# ==============================================================================

input_file <- file.path(
  processed_dir,
  "charls_final_analytic.rds"
)

if (!file.exists(input_file)) {
  stop(
    "Final analytic dataset was not found. ",
    "Run RR/main/02_define_mets_and_complete_cases.R first."
  )
}

analytic_data <- readRDS(input_file)


# Reproducibility checks
stopifnot(
  nrow(analytic_data) == 9867,
  sum(analytic_data$met_diagnosis == 1) == 2197,
  all(complete.cases(analytic_data))
)# ==============================================================================
# 2. Helper functions for Table 1
# ==============================================================================

analytic_data <- analytic_data %>%
  mutate(
    met_status = factor(
      met_diagnosis,
      levels = c(0, 1),
      labels = c("Absence", "Presence")
    )
  )


format_mean_sd <- function(x) {
  sprintf(
    "%.2f (%.2f)",
    mean(x),
    sd(x)
  )
}


format_median_iqr <- function(x) {
  sprintf(
    "%.2f [%.2f, %.2f]",
    median(x),
    quantile(x, 0.25),
    quantile(x, 0.75)
  )
}


format_p <- function(p) {
  if (p < 0.001) {
    "<0.001"
  } else {
    sprintf("%.3f", p)
  }
}


continuous_stats <- function(
    data,
    variable,
    summary_type = c("mean", "median")
) {
  
  summary_type <- match.arg(summary_type)
  
  overall <- data[[variable]]
  absence <- data[[variable]][data$met_status == "Absence"]
  presence <- data[[variable]][data$met_status == "Presence"]
  
  if (summary_type == "mean") {
    
    overall_text <- format_mean_sd(overall)
    absence_text <- format_mean_sd(absence)
    presence_text <- format_mean_sd(presence)
    
    p_value <- t.test(
      absence,
      presence
    )$p.value
    
  } else {
    
    overall_text <- format_median_iqr(overall)
    absence_text <- format_median_iqr(absence)
    presence_text <- format_median_iqr(presence)
    
    p_value <- wilcox.test(
      absence,
      presence
    )$p.value
  }
  
  c(
    overall_text,
    absence_text,
    presence_text,
    format_p(p_value)
  )
}


categorical_stats <- function(
    data,
    variable,
    value
) {
  
  overall_n <- sum(
    data[[variable]] == value
  )
  
  absence_data <- data %>%
    filter(
      met_status == "Absence"
    )
  
  presence_data <- data %>%
    filter(
      met_status == "Presence"
    )
  
  absence_n <- sum(
    absence_data[[variable]] == value
  )
  
  presence_n <- sum(
    presence_data[[variable]] == value
  )
  
  fmt <- function(n, denominator) {
    sprintf(
      "%d (%.1f)",
      n,
      100 * n / denominator
    )
  }
  
  c(
    fmt(overall_n, nrow(data)),
    fmt(absence_n, nrow(absence_data)),
    fmt(presence_n, nrow(presence_data))
  )
}# ==============================================================================
# 3. Create Table 1
# ==============================================================================

table1_rows <- list()

table1_rows[[1]] <- c(
  "n",
  nrow(analytic_data),
  sum(analytic_data$met_status == "Absence"),
  sum(analytic_data$met_status == "Presence"),
  ""
)

table1_rows[[2]] <- c(
  "age (years)",
  continuous_stats(
    analytic_data,
    "age",
    "mean"
  )
)


# Sex
p_sex <- chisq.test(
  table(
    analytic_data$gender,
    analytic_data$met_status
  )
)$p.value

table1_rows[[3]] <- c(
  "sex, n (%)",
  "",
  "",
  "",
  format_p(p_sex)
)

table1_rows[[4]] <- c(
  "  Male",
  categorical_stats(
    analytic_data,
    "gender",
    1
  ),
  ""
)

table1_rows[[5]] <- c(
  "  Female",
  categorical_stats(
    analytic_data,
    "gender",
    2
  ),
  ""
)


# Smoking
p_smoking <- chisq.test(
  table(
    analytic_data$smoking,
    analytic_data$met_status
  )
)$p.value

table1_rows[[6]] <- c(
  "smoking, n (%)",
  "",
  "",
  "",
  format_p(p_smoking)
)

table1_rows[[7]] <- c(
  "  Smoker",
  categorical_stats(
    analytic_data,
    "smoking",
    1
  ),
  ""
)

table1_rows[[8]] <- c(
  "  Non-smoker",
  categorical_stats(
    analytic_data,
    "smoking",
    2
  ),
  ""
)


# Drinking
p_drinking <- chisq.test(
  table(
    analytic_data$drinking,
    analytic_data$met_status
  )
)$p.value

table1_rows[[9]] <- c(
  "drinking, n (%)",
  "",
  "",
  "",
  format_p(p_drinking)
)

table1_rows[[10]] <- c(
  "  Drinker",
  categorical_stats(
    analytic_data,
    "drinking",
    1
  ),
  ""
)

table1_rows[[11]] <- c(
  "  Non-drinker",
  categorical_stats(
    analytic_data,
    "drinking",
    2
  ),
  ""
)
table1_rows[[12]] <- c(
  "BMI (kg/m^2)",
  continuous_stats(analytic_data, "BMI", "mean")
)

table1_rows[[13]] <- c(
  "waist (cm)",
  continuous_stats(analytic_data, "waist", "mean")
)

table1_rows[[14]] <- c(
  "SBP (mmHg)",
  continuous_stats(analytic_data, "SBP", "mean")
)

table1_rows[[15]] <- c(
  "DBP (mmHg)",
  continuous_stats(analytic_data, "DBP", "mean")
)

table1_rows[[16]] <- c(
  "pulse (bpm)",
  continuous_stats(analytic_data, "pulse", "mean")
)

table1_rows[[17]] <- c(
  "grip (kg)",
  continuous_stats(analytic_data, "grip", "mean")
)

table1_rows[[18]] <- c(
  "CRP (mg/L)",
  continuous_stats(analytic_data, "CRP", "median")
)

table1_rows[[19]] <- c(
  "BUN (mg/dL)",
  continuous_stats(analytic_data, "BUN", "median")
)

table1_rows[[20]] <- c(
  "HDL-C (mg/dL)",
  continuous_stats(analytic_data, "HDL_C", "mean")
)

table1_rows[[21]] <- c(
  "LDL-C (mg/dL)",
  continuous_stats(analytic_data, "LDL_C", "mean")
)

table1_rows[[22]] <- c(
  "TC (mg/dL)",
  continuous_stats(analytic_data, "TC", "mean")
)

table1_rows[[23]] <- c(
  "UA (mg/dL)",
  continuous_stats(analytic_data, "UA", "median")
)

table1_rows[[24]] <- c(
  "CREA (mg/dL)",
  continuous_stats(analytic_data, "CREA", "median")
)

table1_rows[[25]] <- c(
  "HbA1c (%)",
  continuous_stats(analytic_data, "HbA1c", "median")
)

table1_rows[[26]] <- c(
  "TG (mg/dL)",
  continuous_stats(analytic_data, "TG", "median")
)

table1_rows[[27]] <- c(
  "FPG (mg/dL)",
  continuous_stats(analytic_data, "FPG", "median")
)


table1 <- as.data.frame(
  do.call(
    rbind,
    table1_rows
  ),
  stringsAsFactors = FALSE
)

colnames(table1) <- c(
  "Characteristics",
  "Overall",
  "Absence",
  "Presence",
  "P_value"
)


write.csv(
  table1,
  file.path(
    results_dir,
    "Table1_baseline_characteristics.csv"
  ),
  row.names = FALSE
)# ==============================================================================
# 4. Recode variables for model development
# ==============================================================================

model_data <- analytic_data %>%
  select(
    -met_status
  ) %>%
  mutate(
    
    # sex:
    # 0 = Male
    # 1 = Female
    gender = ifelse(
      gender == 2,
      1,
      0
    ),
    
    # smoking:
    # 0 = Non-smoker
    # 1 = Smoker
    smoking = ifelse(
      smoking == 1,
      1,
      0
    ),
    
    # drinking:
    # 0 = Non-drinker
    # 1 = Drinker
    drinking = ifelse(
      drinking == 1,
      1,
      0
    ),
    
    # Outcome for caret-based model development
    met_diagnosis = factor(
      met_diagnosis,
      levels = c(0, 1)
    )
  )# ==============================================================================
# 5. Stratified 70/30 training/testing split
# ==============================================================================

set.seed(SEED)

train_index <- createDataPartition(
  model_data$met_diagnosis,
  p = 0.70,
  list = FALSE
)

train_set <- model_data[
  train_index,
]

test_set <- model_data[
  -train_index,
]# ==============================================================================
# 6. Verify the split against the final manuscript
# ==============================================================================

train_cases <- sum(
  train_set$met_diagnosis == "1"
)

test_cases <- sum(
  test_set$met_diagnosis == "1"
)

stopifnot(
  nrow(train_set) == 6907,
  nrow(test_set) == 2960,
  train_cases == 1538,
  test_cases == 659,
  nrow(train_set) + nrow(test_set) == 9867
)# ==============================================================================
# 7. Save training and testing datasets
# ==============================================================================

saveRDS(
  train_set,
  file.path(
    processed_dir,
    "train_set.rds"
  )
)

saveRDS(
  test_set,
  file.path(
    processed_dir,
    "test_set.rds"
  )
)# ==============================================================================
# 8. Helper functions for descriptive Table 2
# ==============================================================================

format_dataset_mean <- function(
    data,
    variable
) {
  sprintf(
    "%.2f (%.2f)",
    mean(data[[variable]]),
    sd(data[[variable]])
  )
}


format_dataset_median <- function(
    data,
    variable
) {
  sprintf(
    "%.2f [%.2f, %.2f]",
    median(data[[variable]]),
    quantile(data[[variable]], 0.25),
    quantile(data[[variable]], 0.75)
  )
}


format_dataset_category <- function(
    data,
    variable,
    value
) {
  
  n_value <- sum(
    data[[variable]] == value
  )
  
  sprintf(
    "%d (%.1f)",
    n_value,
    100 * n_value / nrow(data)
  )
}# ==============================================================================
# 9. Create descriptive Table 2
# ==============================================================================

table2_rows <- list()

table2_rows[[1]] <- c(
  "n",
  nrow(train_set),
  nrow(test_set)
)

table2_rows[[2]] <- c(
  "MetS, n (%)",
  format_dataset_category(
    train_set,
    "met_diagnosis",
    "1"
  ),
  format_dataset_category(
    test_set,
    "met_diagnosis",
    "1"
  )
)

table2_rows[[3]] <- c(
  "age (years)",
  format_dataset_mean(train_set, "age"),
  format_dataset_mean(test_set, "age")
)


# Sex
table2_rows[[4]] <- c(
  "sex, n (%)",
  "",
  ""
)

table2_rows[[5]] <- c(
  "  Female",
  format_dataset_category(train_set, "gender", 1),
  format_dataset_category(test_set, "gender", 1)
)

table2_rows[[6]] <- c(
  "  Male",
  format_dataset_category(train_set, "gender", 0),
  format_dataset_category(test_set, "gender", 0)
)


# Smoking
table2_rows[[7]] <- c(
  "smoking, n (%)",
  "",
  ""
)

table2_rows[[8]] <- c(
  "  Smoker",
  format_dataset_category(train_set, "smoking", 1),
  format_dataset_category(test_set, "smoking", 1)
)

table2_rows[[9]] <- c(
  "  Non-smoker",
  format_dataset_category(train_set, "smoking", 0),
  format_dataset_category(test_set, "smoking", 0)
)


# Drinking
table2_rows[[10]] <- c(
  "drinking, n (%)",
  "",
  ""
)

table2_rows[[11]] <- c(
  "  Drinker",
  format_dataset_category(train_set, "drinking", 1),
  format_dataset_category(test_set, "drinking", 1)
)

table2_rows[[12]] <- c(
  "  Non-drinker",
  format_dataset_category(train_set, "drinking", 0),
  format_dataset_category(test_set, "drinking", 0)
)
table2_rows[[13]] <- c(
  "BMI (kg/m^2)",
  format_dataset_mean(train_set, "BMI"),
  format_dataset_mean(test_set, "BMI")
)

table2_rows[[14]] <- c(
  "waist (cm)",
  format_dataset_mean(train_set, "waist"),
  format_dataset_mean(test_set, "waist")
)

table2_rows[[15]] <- c(
  "SBP (mmHg)",
  format_dataset_mean(train_set, "SBP"),
  format_dataset_mean(test_set, "SBP")
)

table2_rows[[16]] <- c(
  "DBP (mmHg)",
  format_dataset_mean(train_set, "DBP"),
  format_dataset_mean(test_set, "DBP")
)

table2_rows[[17]] <- c(
  "pulse (bpm)",
  format_dataset_mean(train_set, "pulse"),
  format_dataset_mean(test_set, "pulse")
)

table2_rows[[18]] <- c(
  "grip (kg)",
  format_dataset_mean(train_set, "grip"),
  format_dataset_mean(test_set, "grip")
)

table2_rows[[19]] <- c(
  "CRP (mg/L)",
  format_dataset_median(train_set, "CRP"),
  format_dataset_median(test_set, "CRP")
)

table2_rows[[20]] <- c(
  "BUN (mg/dL)",
  format_dataset_median(train_set, "BUN"),
  format_dataset_median(test_set, "BUN")
)

table2_rows[[21]] <- c(
  "HDL-C (mg/dL)",
  format_dataset_mean(train_set, "HDL_C"),
  format_dataset_mean(test_set, "HDL_C")
)

table2_rows[[22]] <- c(
  "LDL-C (mg/dL)",
  format_dataset_mean(train_set, "LDL_C"),
  format_dataset_mean(test_set, "LDL_C")
)

table2_rows[[23]] <- c(
  "TC (mg/dL)",
  format_dataset_mean(train_set, "TC"),
  format_dataset_mean(test_set, "TC")
)

table2_rows[[24]] <- c(
  "UA (mg/dL)",
  format_dataset_median(train_set, "UA"),
  format_dataset_median(test_set, "UA")
)

table2_rows[[25]] <- c(
  "CREA (mg/dL)",
  format_dataset_median(train_set, "CREA"),
  format_dataset_median(test_set, "CREA")
)

table2_rows[[26]] <- c(
  "HbA1c (%)",
  format_dataset_median(train_set, "HbA1c"),
  format_dataset_median(test_set, "HbA1c")
)

table2_rows[[27]] <- c(
  "TG (mg/dL)",
  format_dataset_median(train_set, "TG"),
  format_dataset_median(test_set, "TG")
)

table2_rows[[28]] <- c(
  "FPG (mg/dL)",
  format_dataset_median(train_set, "FPG"),
  format_dataset_median(test_set, "FPG")
)


table2 <- as.data.frame(
  do.call(
    rbind,
    table2_rows
  ),
  stringsAsFactors = FALSE
)

colnames(table2) <- c(
  "Variables",
  "Training_Set",
  "Testing_Set"
)


write.csv(
  table2,
  file.path(
    results_dir,
    "Table2_train_test_characteristics.csv"
  ),
  row.names = FALSE
)# ==============================================================================
# 10. Save split summary
# ==============================================================================

split_summary <- data.frame(
  Dataset = c(
    "Overall",
    "Training",
    "Testing"
  ),
  
  N = c(
    nrow(model_data),
    nrow(train_set),
    nrow(test_set)
  ),
  
  MetS_cases = c(
    sum(model_data$met_diagnosis == "1"),
    train_cases,
    test_cases
  ),
  
  MetS_prevalence = c(
    mean(model_data$met_diagnosis == "1"),
    mean(train_set$met_diagnosis == "1"),
    mean(test_set$met_diagnosis == "1")
  )
)

write.csv(
  split_summary,
  file.path(
    results_dir,
    "train_test_split_summary.csv"
  ),
  row.names = FALSE
)