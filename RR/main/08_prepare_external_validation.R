# ==============================================================================
# 08_prepare_external_validation.R
# Prepare the independent external-validation cohort
#
# IMPORTANT:
# - External raw hospital data are private and are NOT distributed.
# - No outcome-based sampling is performed.
# - Only variables required for external validation are retained.
# ==============================================================================

library(dplyr)
library(tidyr)
library(readxl)
library(readr)

source("RR/main/00_setup.R")


# ==============================================================================
# 1. Locate external-validation source files
# ==============================================================================

if (!dir.exists(external_raw_dir)) {
  stop(
    "External-validation raw-data directory was not found: ",
    external_raw_dir
  )
}


external_files <- list.files(
  external_raw_dir,
  pattern = "\\.xls$",
  full.names = TRUE
)


if (length(external_files) == 0) {
  stop(
    "No .xls files were found in the external-validation raw-data directory."
  )
}


cat(
  "Number of external-validation source files:",
  length(external_files),
  "\n"
)
# ==============================================================================
# 2. Define required external-validation variables
# ==============================================================================

required_external_variables <- c(
  "年龄",
  "性别",
  "体重指数",
  "脉搏(次/分)",
  "尿素",
  "总胆固醇",
  "尿酸",
  "糖化血红蛋白(HPLC)",
  "腰围(cm)",
  "收缩压(mmHg)",
  "舒张压(mmHg)",
  "葡萄糖",
  "甘油三酯",
  "高密度脂蛋白胆固醇"
)
# ==============================================================================
# 3. Read and merge all external-validation files
# ==============================================================================

external_list <- lapply(
  external_files,
  function(file_path) {
    
    df <- suppressWarnings(
      read_excel(file_path)
    )
    
    missing_vars <- setdiff(
      required_external_variables,
      names(df)
    )
    
    if (length(missing_vars) > 0) {
      stop(
        "Required variables are missing from an external-validation file: ",
        paste(
          missing_vars,
          collapse = ", "
        )
      )
    }
    
    
    # Retain only variables required for the analysis.
    df <- df %>%
      select(
        all_of(
          required_external_variables
        )
      )
    
    
    # Variables required for MetS ascertainment
    # are converted safely to numeric.
    df <- df %>%
      mutate(
        across(
          c(
            "葡萄糖",
            "收缩压(mmHg)",
            "舒张压(mmHg)",
            "甘油三酯",
            "高密度脂蛋白胆固醇",
            "腰围(cm)"
          ),
          ~ suppressWarnings(
            as.numeric(
              as.character(.x)
            )
          )
        )
      )
    
    
    df
  }
)


data_merged <- bind_rows(
  external_list
)


cat(
  "Merged external-validation records:",
  nrow(data_merged),
  "\n"
)
# ==============================================================================
# 4. Retain records with complete variables required for MetS ascertainment
# ==============================================================================

data_diagnosable <- data_merged %>%
  drop_na(
    性别,
    `腰围(cm)`,
    葡萄糖,
    `收缩压(mmHg)`,
    `舒张压(mmHg)`,
    甘油三酯,
    高密度脂蛋白胆固醇
  )


cat(
  "Records with complete MetS diagnostic components:",
  nrow(data_diagnosable),
  "\n"
)
# ==============================================================================
# 5. Ascertain MetS in the external cohort
# ==============================================================================

data_with_mets <- data_diagnosable %>%
  mutate(
    
    Crit_Obesity = ifelse(
      (
        性别 == "男" &
          `腰围(cm)` >= 90
      ) |
        (
          性别 == "女" &
            `腰围(cm)` >= 85
        ),
      1,
      0
    ),
    
    Crit_Glucose = ifelse(
      葡萄糖 >= 6.1,
      1,
      0
    ),
    
    Crit_BP = ifelse(
      `收缩压(mmHg)` >= 130 |
        `舒张压(mmHg)` >= 85,
      1,
      0
    ),
    
    Crit_TG = ifelse(
      甘油三酯 >= 1.70,
      1,
      0
    ),
    
    Crit_HDL = ifelse(
      高密度脂蛋白胆固醇 < 1.04,
      1,
      0
    ),
    
    MetS_Score =
      Crit_Obesity +
      Crit_Glucose +
      Crit_BP +
      Crit_TG +
      Crit_HDL,
    
    MetS_Status = factor(
      ifelse(
        MetS_Score >= 3,
        1,
        0
      ),
      levels = c(0, 1),
      labels = c(
        "No",
        "Yes"
      )
    )
  )
# ==============================================================================
# 6. Parse age and harmonize predictor units
# ==============================================================================

data_prepared <- data_with_mets %>%
  mutate(
    
    # Example: "61岁" -> 61
    年龄 = parse_number(
      as.character(年龄)
    ),
    
    体重指数 = suppressWarnings(
      as.numeric(
        as.character(体重指数)
      )
    ),
    
    `脉搏(次/分)` = suppressWarnings(
      as.numeric(
        as.character(
          `脉搏(次/分)`
        )
      )
    ),
    
    尿素 = suppressWarnings(
      as.numeric(
        as.character(尿素)
      )
    ),
    
    总胆固醇 = suppressWarnings(
      as.numeric(
        as.character(总胆固醇)
      )
    ),
    
    尿酸 = suppressWarnings(
      as.numeric(
        as.character(尿酸)
      )
    ),
    
    `糖化血红蛋白(HPLC)` =
      suppressWarnings(
        as.numeric(
          as.character(
            `糖化血红蛋白(HPLC)`
          )
        )
      ),
    
    # Harmonize units with the CHARLS development dataset:
    # UA: µmol/L -> mg/dL
    尿酸 = 尿酸 / 59.48,
    
    # TC: mmol/L -> mg/dL
    总胆固醇 =
      总胆固醇 * 38.67,
    
    # BUN: mmol/L -> mg/dL
    尿素 =
      尿素 * 2.8
  )
# ==============================================================================
# 7. Apply eligibility and completeness criteria
# ==============================================================================

external_final <- data_prepared %>%
  
  filter(
    年龄 >= 45,
    `腰围(cm)` >= 40,
    `腰围(cm)` <= 200
  ) %>%
  
  drop_na(
    年龄,
    体重指数,
    `脉搏(次/分)`,
    尿素,
    总胆固醇,
    尿酸,
    `糖化血红蛋白(HPLC)`,
    MetS_Status
  )
# ==============================================================================
# 8. Create the model-input dataset
# ==============================================================================

external_model_data <- external_final %>%
  transmute(
    age =
      as.numeric(年龄),
    
    BMI =
      as.numeric(体重指数),
    
    pulse =
      as.numeric(
        `脉搏(次/分)`
      ),
    
    BUN =
      as.numeric(尿素),
    
    TC =
      as.numeric(总胆固醇),
    
    UA =
      as.numeric(尿酸),
    
    HbA1c =
      as.numeric(
        `糖化血红蛋白(HPLC)`
      ),
    
    met_diagnosis =
      ifelse(
        MetS_Status == "Yes",
        1L,
        0L
      )
  )
# ==============================================================================
# 9. Reproducibility checks
# ==============================================================================

n_external <- nrow(
  external_model_data
)

n_mets_external <- sum(
  external_model_data$met_diagnosis == 1
)

n_nonmets_external <- sum(
  external_model_data$met_diagnosis == 0
)


stopifnot(
  n_external == 553,
  n_mets_external == 213,
  n_nonmets_external == 340,
  all(
    complete.cases(
      external_model_data
    )
  ),
  identical(
    names(
      external_model_data
    )[1:7],
    final_predictors
  )
)
# ==============================================================================
# 10. Save local processed data
# ==============================================================================

saveRDS(
  external_final,
  file.path(
    processed_dir,
    "external_validation_full.rds"
  )
)


saveRDS(
  external_model_data,
  file.path(
    processed_dir,
    "external_validation_model_data.rds"
  )
)
# ==============================================================================
# 11. Save non-individual summary
# ==============================================================================

external_preparation_summary <- data.frame(
  
  Dataset = "External validation",
  
  N = n_external,
  
  MetS_cases =
    n_mets_external,
  
  Non_MetS =
    n_nonmets_external,
  
  MetS_prevalence =
    n_mets_external /
    n_external
)


write.csv(
  external_preparation_summary,
  file.path(
    results_dir,
    "external_validation_preparation_summary.csv"
  ),
  row.names = FALSE
)
# ==============================================================================
# 12. Reproducibility summary
# ==============================================================================

cat("\n")
cat("============================================================\n")
cat("External-validation data preparation summary\n")
cat("============================================================\n")

cat(
  "Final external-validation sample =",
  n_external,
  "\n"
)

cat(
  "MetS cases =",
  n_mets_external,
  "\n"
)

cat(
  "Non-MetS =",
  n_nonmets_external,
  "\n"
)

cat(
  "MetS prevalence =",
  sprintf(
    "%.1f%%",
    100 *
      n_mets_external /
      n_external
  ),
  "\n"
)

cat(
  "Outcome-based sampling performed: No\n"
)

cat(
  "Model predictors:",
  paste(
    final_predictors,
    collapse = ", "
  ),
  "\n"
)

cat("============================================================\n")