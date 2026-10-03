# ==============================================================================
# 01_prepare_charls_data.R
# Prepare CHARLS 2015 variables required for MetS ascertainment and modeling
# ==============================================================================

library(haven)
library(dplyr)
library(tidyr)

source("RR/main/00_setup.R")


# ==============================================================================
# 1. File paths
# ==============================================================================

demographic_file <- file.path(
  charls_raw_dir,
  "Demographic_Background.dta"
)

health_file <- file.path(
  charls_raw_dir,
  "Health_Status_and_Functioning.dta"
)

biomarker_file <- file.path(
  charls_raw_dir,
  "Biomarker.dta"
)

blood_file <- file.path(
  charls_raw_dir,
  "Blood.dta"
)

required_files <- c(
  demographic_file,
  health_file,
  biomarker_file,
  blood_file
)

if (any(!file.exists(required_files))) {
  stop(
    "One or more required CHARLS 2015 raw data files were not found in: ",
    charls_raw_dir
  )
}


# ==============================================================================
# 2. Demographic variables
# ==============================================================================

demographic_raw <- read_dta(demographic_file)

demographic <- demographic_raw %>%
  transmute(
    ID = as.character(ID),
    birth_year = as.numeric(ba004_w3_1),
    birth_month = as.numeric(ba004_w3_2),
    birth_day = as.numeric(ba004_w3_3),
    gender = as.numeric(ba000_w2_3)
  ) %>%
  mutate(
    birth_year_clean = case_when(
      birth_year >= 1900 & birth_year <= 2015 ~ birth_year,
      TRUE ~ NA_real_
    ),
    birth_month_clean = case_when(
      birth_month >= 1 & birth_month <= 12 ~ birth_month,
      TRUE ~ 7
    ),
    birth_day_clean = case_when(
      birth_day >= 1 & birth_day <= 31 ~ birth_day,
      TRUE ~ 15
    ),
    birth_day_adj = case_when(
      birth_month_clean == 2 & birth_day_clean > 29 ~ 28,
      birth_month_clean %in% c(4, 6, 9, 11) &
        birth_day_clean > 30 ~ 30,
      TRUE ~ birth_day_clean
    ),
    age = case_when(
      is.na(birth_year_clean) ~ NA_real_,
      (7 > birth_month_clean) |
        (7 == birth_month_clean & 1 >= birth_day_adj) ~
        2015 - birth_year_clean,
      TRUE ~ 2015 - birth_year_clean - 1
    )
  ) %>%
  select(
    ID,
    age,
    gender
  )


# ==============================================================================
# 3. Smoking and drinking
# ==============================================================================

health_raw <- read_dta(health_file)

health <- health_raw %>%
  transmute(
    ID = as.character(ID),
    DA059 = da059,
    DA061 = da061,
    drinking_raw = da067
  ) %>%
  mutate(
    smoking = case_when(
      DA061 == 1 ~ 1,
      DA061 %in% c(2, 3) ~ 2,
      is.na(DA061) & DA059 == 2 ~ 2,
      TRUE ~ NA_real_
    ),
    drinking = case_when(
      drinking_raw %in% c(1, 2) ~ 1,
      drinking_raw == 3 ~ 2,
      TRUE ~ NA_real_
    )
  ) %>%
  select(
    ID,
    smoking,
    drinking
  )


# ==============================================================================
# 4. Anthropometric and physical measurements
# ==============================================================================

biomarker_raw <- read_dta(biomarker_file)

biomarker <- biomarker_raw %>%
  transmute(
    ID = as.character(ID),
    height = qi002,
    weight = ql002,
    waist = qm002,
    
    SBP1 = qa003,
    DBP1 = qa004,
    pulse1 = qa005,
    
    SBP2 = qa007,
    DBP2 = qa008,
    pulse2 = qa009,
    
    SBP3 = qa011,
    DBP3 = qa012,
    pulse3 = qa013,
    
    grip_left1 = qc003,
    grip_right1 = qc004,
    grip_left2 = qc005,
    grip_right2 = qc006
  ) %>%
  mutate(
    height_m = height / 100,
    
    BMI = weight / (height_m^2),
    
    SBP = rowMeans(
      cbind(SBP1, SBP2, SBP3),
      na.rm = TRUE
    ),
    
    DBP = rowMeans(
      cbind(DBP1, DBP2, DBP3),
      na.rm = TRUE
    ),
    
    pulse = rowMeans(
      cbind(pulse1, pulse2, pulse3),
      na.rm = TRUE
    ),
    
    grip_left_mean = rowMeans(
      cbind(grip_left1, grip_left2),
      na.rm = TRUE
    ),
    
    grip_right_mean = rowMeans(
      cbind(grip_right1, grip_right2),
      na.rm = TRUE
    ),
    
    grip = pmax(
      grip_left_mean,
      grip_right_mean,
      na.rm = TRUE
    )
  ) %>%
  mutate(
    across(
      c(BMI, SBP, DBP, pulse, grip),
      ~ ifelse(is.nan(.x), NA_real_, .x)
    )
  ) %>%
  select(
    ID,
    BMI,
    waist,
    SBP,
    DBP,
    pulse,
    grip
  )


# ==============================================================================
# 5. Blood biomarkers
# ==============================================================================

blood_raw <- read_dta(blood_file)

blood <- blood_raw %>%
  transmute(
    ID = as.character(ID),
    CRP = bl_crp,
    BUN = bl_bun,
    HDL_C = bl_hdl,
    LDL_C = bl_ldl,
    TC = bl_cho,
    UA = bl_ua,
    CREA = bl_crea,
    HbA1c = bl_hbalc,
    TG = bl_tg,
    FPG_raw = bl_glu,
    fasting = as.numeric(bl_fasting)
  ) %>%
  mutate(
    FPG = ifelse(
      fasting == 1,
      FPG_raw,
      NA_real_
    )
  ) %>%
  select(
    ID,
    CRP,
    BUN,
    HDL_C,
    LDL_C,
    TC,
    UA,
    CREA,
    HbA1c,
    TG,
    FPG
  )# ==============================================================================
# 6. Hypertension and diabetes treatment status
# ==============================================================================

treatment <- health_raw %>%
  transmute(
    ID = as.character(ID),
    
    hypertension_diagnosis = da007_1_,
    hypertension_treatment_chinese = da011s1,
    hypertension_treatment_western = da011s2,
    hypertension_treatment_none = da011s3,
    
    diabetes_diagnosis = da007_3_,
    diabetes_treatment_chinese = da014s1,
    diabetes_treatment_western = da014s2,
    diabetes_treatment_insulin = da014s3,
    diabetes_treatment_none = da014s4
  ) %>%
  mutate(
    hypertension_diagnosed = case_when(
      hypertension_diagnosis == 1 ~ 1,
      hypertension_diagnosis == 2 ~ 0,
      TRUE ~ NA_real_
    ),
    
    diabetes_diagnosed = case_when(
      diabetes_diagnosis == 1 ~ 1,
      diabetes_diagnosis == 2 ~ 0,
      TRUE ~ NA_real_
    ),
    
    hypertension_treated = case_when(
      hypertension_diagnosed == 1 &
        (
          hypertension_treatment_chinese == 1 |
            hypertension_treatment_western == 2
        ) ~ 1,
      
      hypertension_diagnosed == 1 &
        hypertension_treatment_none == 3 ~ 0,
      
      hypertension_diagnosed == 0 ~ 0,
      
      TRUE ~ NA_real_
    ),
    
    diabetes_treated = case_when(
      diabetes_diagnosed == 1 &
        (
          diabetes_treatment_chinese == 1 |
            diabetes_treatment_western == 2 |
            diabetes_treatment_insulin == 3
        ) ~ 1,
      
      diabetes_diagnosed == 1 &
        diabetes_treatment_none == 4 ~ 0,
      
      diabetes_diagnosed == 0 ~ 0,
      
      TRUE ~ NA_real_
    )
  ) %>%
  select(
    ID,
    hypertension_treated,
    diabetes_treated
  )# ==============================================================================
# 7. Merge CHARLS modules
# ==============================================================================

charls_merged <- demographic %>%
  inner_join(
    health,
    by = "ID"
  ) %>%
  inner_join(
    biomarker,
    by = "ID"
  ) %>%
  inner_join(
    blood,
    by = "ID"
  ) %>%
  inner_join(
    treatment,
    by = "ID"
  )# ==============================================================================
# 8. Retain participants with sufficient information for MetS ascertainment
# ==============================================================================

charls_prepared <- charls_merged %>%
  filter(
    !(is.na(FPG) & is.na(diabetes_treated)),
    !(is.na(SBP) & is.na(DBP) & is.na(hypertension_treated)),
    !is.na(TG),
    !is.na(HDL_C),
    !is.na(gender),
    !is.na(waist)
  )


# ==============================================================================
# 9. Basic integrity checks
# ==============================================================================

stopifnot(
  !anyDuplicated(charls_prepared$ID),
  
  all(
    na.omit(unique(charls_prepared$gender)) %in%
      c(1, 2)
  ),
  
  all(
    na.omit(unique(charls_prepared$smoking)) %in%
      c(1, 2)
  ),
  
  all(
    na.omit(unique(charls_prepared$drinking)) %in%
      c(1, 2)
  ),
  
  all(
    na.omit(unique(charls_prepared$hypertension_treated)) %in%
      c(0, 1)
  ),
  
  all(
    na.omit(unique(charls_prepared$diabetes_treated)) %in%
      c(0, 1)
  )
)


# ==============================================================================
# 10. Save prepared CHARLS data
# ==============================================================================

saveRDS(
  charls_prepared,
  file.path(
    processed_dir,
    "charls_prepared_for_mets.rds"
  )
)