# ==============================================================================
# 02_define_mets_and_complete_cases.R
# Define MetS and create the final complete-case CHARLS analytic dataset
# ==============================================================================

library(dplyr)
library(tidyr)
library(haven)

source("RR/main/00_setup.R")


# ==============================================================================
# 1. Load prepared CHARLS data
# ==============================================================================

input_file <- file.path(
  processed_dir,
  "charls_prepared_for_mets.rds"
)

if (!file.exists(input_file)) {
  stop(
    "Prepared CHARLS data were not found. ",
    "Run RR/main/01_prepare_charls_data.R first."
  )
}

df <- readRDS(input_file) %>%
  zap_labels()


# ==============================================================================
# 2. Convert diagnostic laboratory variables to mmol/L for MetS ascertainment
# ==============================================================================

# The original CHARLS blood variables used in this analysis were converted
# to mmol/L before applying the MetS diagnostic criteria.

if (mean(df$FPG, na.rm = TRUE) > 20) {
  df$FPG <- df$FPG / 18.01
}

if (mean(df$TG, na.rm = TRUE) > 10) {
  df$TG <- df$TG / 88.57
}

if (mean(df$HDL_C, na.rm = TRUE) > 2) {
  df$HDL_C <- df$HDL_C / 38.67
}


# ==============================================================================
# 3. Ensure that the hyperglycemia component can be ascertained
# ==============================================================================

# Participants with missing FPG were retained only when diabetes treatment
# status itself established the hyperglycemia component.

df <- df %>%
  filter(
    !(
      is.na(FPG) &
        coalesce(diabetes_treated, 0) != 1
    )
  )


# ==============================================================================
# 4. Define the five MetS components
# ==============================================================================

# MetS was defined according to the Chinese 2020 guideline.
# Presence of at least three of the five components indicates MetS.

df <- df %>%
  mutate(
    
    # Abdominal obesity
    met_obesity = case_when(
      gender == 1 & waist >= 90 ~ 1,
      gender == 2 & waist >= 85 ~ 1,
      TRUE ~ 0
    ),
    
    # Hyperglycemia
    met_hyperglycemia = if_else(
      FPG >= 6.1 |
        coalesce(diabetes_treated, 0) == 1,
      1,
      0
    ),
    
    # Hypertension
    met_hypertension = if_else(
      SBP >= 130 |
        DBP >= 85 |
        coalesce(hypertension_treated, 0) == 1,
      1,
      0
    ),
    
    # Hypertriglyceridemia
    met_tg = if_else(
      TG >= 1.70,
      1,
      0
    ),
    
    # Low HDL-C
    met_hdlc = if_else(
      HDL_C < 1.04,
      1,
      0
    )
  )# ==============================================================================
# 5. Create the MetS outcome
# ==============================================================================

df <- df %>%
  mutate(
    n_components =
      met_obesity +
      met_hyperglycemia +
      met_hypertension +
      met_tg +
      met_hdlc,
    
    met_diagnosis = if_else(
      n_components >= 3,
      1,
      0
    )
  )# ==============================================================================
# 6. Retain variables required for the analytic dataset
# ==============================================================================

analytic_variables <- c(
  "ID",
  "age",
  "gender",
  "smoking",
  "drinking",
  "BMI",
  "waist",
  "SBP",
  "DBP",
  "pulse",
  "grip",
  "CRP",
  "BUN",
  "HDL_C",
  "LDL_C",
  "TC",
  "UA",
  "CREA",
  "HbA1c",
  "TG",
  "FPG",
  "met_diagnosis"
)

missing_variables <- setdiff(
  analytic_variables,
  names(df)
)

if (length(missing_variables) > 0) {
  stop(
    "Missing required variables: ",
    paste(missing_variables, collapse = ", ")
  )
}

df <- df %>%
  select(
    all_of(analytic_variables)
  )# ==============================================================================
# 7. Recode nonmeasurement values
# ==============================================================================

df <- df %>%
  mutate(
    across(
      where(is.numeric),
      ~ ifelse(
        .x %in% c(993, 999),
        NA,
        .x
      )
    )
  )


# ==============================================================================
# 8. Restrict to participants aged 45 years or older
# ==============================================================================

df <- df %>%
  filter(
    age >= 45
  )


# ==============================================================================
# 9. Apply prespecified physiological plausibility ranges
# ==============================================================================

# Missing values are retained at this stage and handled subsequently by
# complete-case analysis.
# FPG and TG are still expressed in mmol/L.

df <- df %>%
  filter(
    is.na(age)   | age <= 110,
    is.na(BMI)   | between(BMI, 13, 60),
    is.na(waist) | between(waist, 40, 200),
    is.na(SBP)   | between(SBP, 70, 300),
    is.na(DBP)   | between(DBP, 30, 200),
    is.na(pulse) | between(pulse, 30, 200),
    is.na(FPG)   | between(FPG, 2.8, 22.8),
    is.na(HbA1c) | between(HbA1c, 3, 25),
    is.na(TG)    | between(TG, 0.2, 100),
    is.na(grip)  | between(grip, 5, 100),
    is.na(CREA)  | CREA <= 5.0
  )# ==============================================================================
# 10. Verify the cohort size before complete-case exclusion
# ==============================================================================

n_before_complete_case <- nrow(df)

stopifnot(
  n_before_complete_case == 10097
)# ==============================================================================
# 11. Convert laboratory variables back to analysis units
# ==============================================================================

df <- df %>%
  mutate(
    FPG = FPG * 18.01,
    HDL_C = HDL_C * 38.67,
    TG = TG * 88.57
  )# ==============================================================================
# 12. Complete-case analysis
# ==============================================================================

final_data <- df[
  complete.cases(df),
]

n_removed_missing <- n_before_complete_case - nrow(final_data)

missing_rate <- (
  n_removed_missing /
    n_before_complete_case
) * 100# ==============================================================================
# 13. Reproducibility checks against the final manuscript
# ==============================================================================

stopifnot(
  n_before_complete_case == 10097,
  n_removed_missing == 230,
  nrow(final_data) == 9867,
  sum(final_data$met_diagnosis == 1) == 2197,
  all(
    complete.cases(final_data)
  )
)# ==============================================================================
# 14. Save final analytic dataset
# ==============================================================================

saveRDS(
  final_data,
  file.path(
    processed_dir,
    "charls_final_analytic.rds"
  )
)


# ==============================================================================
# 15. Save preprocessing summary
# ==============================================================================

preprocessing_summary <- data.frame(
  Stage = c(
    "Eligible before complete-case analysis",
    "Excluded because of missing data",
    "Final analytic cohort",
    "MetS cases in final analytic cohort"
  ),
  N = c(
    n_before_complete_case,
    n_removed_missing,
    nrow(final_data),
    sum(final_data$met_diagnosis == 1)
  )
)

write.csv(
  preprocessing_summary,
  file.path(
    results_dir,
    "preprocessing_summary.csv"
  ),
  row.names = FALSE
)