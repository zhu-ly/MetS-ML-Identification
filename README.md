# Metabolic Syndrome Identification Using Machine Learning

This repository contains the analysis code, locked final CatBoost model, sensitivity analyses, external-validation workflow, synthetic prediction example, and Shiny application source code associated with the manuscript:

**Development and Validation of a Machine Learning-Based Model for Identifying Metabolic Syndrome in Middle-Aged and Older Adults**

The study developed and externally validated machine-learning models for identifying prevalent metabolic syndrome (MetS) among middle-aged and older adults.

Because the study used a cross-sectional design, the model estimates the probability of **prevalent MetS at the time of assessment**. It is not intended to predict future incident MetS.


## Repository overview

The repository is organized as follows:

```text
.
├── README.md
├── RR/
│   ├── main/
│   │   ├── 00_setup.R
│   │   ├── 01_prepare_charls_data.R
│   │   ├── 02_define_mets_and_complete_cases.R
│   │   ├── 03_descriptive_analysis_and_split.R
│   │   ├── 04_feature_selection.R
│   │   ├── 05_train_models.R
│   │   ├── 06_evaluate_models.R
│   │   ├── 07_shap_analysis.R
│   │   ├── 08_prepare_external_validation.R
│   │   └── 09_external_validation.R
│   └── sensitivity/
│       ├── 01_diagnostic_overlap_sensitivity.R
│       ├── 02_lr_standardization_sensitivity.R
│       ├── 03_threshold_sensitivity.R
│       ├── 04_probability_stratification_sensitivity.R
│       ├── 05_shap_zero_crossing_bootstrap.R
│       └── 06_sex_and_age_subgroup_analysis.R
├── data/
│   ├── raw/
│   └── processed/
├── results/
├── model/
│   └── locked/
│       ├── model_cat.rds
│       └── model_cat.cbm
├── app/
│   └── app.R
├── example/
│   ├── 00_create_example_input.R
│   ├── example_input.csv
│   ├── predict_example.R
│   └── 01_verify_native_catboost_model.R
└── environment/
    ├── 00_capture_environment.R
    ├── package_versions.csv
    └── sessionInfo.txt
```

Raw and participant-level processed datasets are intentionally excluded from the public repository.


## Study design and data sources

The development dataset was derived from the **2015 China Health and Retirement Longitudinal Study (CHARLS)**.

Participants aged 45 years or older were included. After sequential eligibility screening and exclusion of values outside prespecified plausibility ranges, 10,097 participants remained. Of these, 230 participants (2.28%) had missing data for one or more variables required for final model development and were excluded using complete-case analysis, resulting in a final analytic cohort of 9,867 participants.

The final analytic cohort was divided into training and testing sets using a stratified 70:30 split with random seed 123:

- Training set: n = 6,907; MetS cases = 1,538
- Testing set: n = 2,960; MetS cases = 659

External validation was performed using data from the Health Management Center of the First Affiliated Hospital of Jinzhou Medical University. The external-validation cohort included 553 participants, of whom 213 had MetS.

The external-validation dataset contains individual-level clinical information and is not included in this repository.


## Metabolic syndrome definition

MetS was defined according to the diagnostic criteria and operational definitions described in the study manuscript. In the CHARLS development dataset, a participant was classified as having MetS when at least three of the following five components were present:

1. Central obesity based on waist circumference;
2. Elevated fasting plasma glucose or diabetes treatment;
3. Elevated blood pressure or antihypertensive treatment;
4. Elevated triglycerides;
5. Reduced high-density lipoprotein cholesterol.
In the external-validation cohort, MetS was operationalized using the available measured components. Information on previously diagnosed or treated diabetes and hypertension was not available in the external dataset and therefore could not be incorporated into external MetS ascertainment.

The detailed operational definitions and data-processing logic are implemented in:

```text
RR/main/01_prepare_charls_data.R
RR/main/02_define_mets_and_complete_cases.R
```


## Predictor selection

Twenty candidate variables were initially considered.

To reduce direct diagnostic overlap between predictors and the MetS outcome definition, six variables that were direct components of the MetS diagnostic criteria were excluded before formal predictor selection:

- waist circumference
- systolic blood pressure
- diastolic blood pressure
- high-density lipoprotein cholesterol
- triglycerides
- fasting plasma glucose

The remaining 14 candidate predictors entered least absolute shrinkage and selection operator (LASSO) analysis.

Using 10-fold cross-validation and the 1-standard-error criterion, eight variables had non-zero coefficients:

- age
- body mass index (BMI)
- pulse
- blood urea nitrogen (BUN)
- low-density lipoprotein cholesterol (LDL-C)
- total cholesterol (TC)
- uric acid (UA)
- glycated hemoglobin (HbA1c)

Because LDL-C showed high collinearity with TC, LDL-C was removed and TC was retained.

The final seven predictors were:

1. age
2. BMI
3. pulse
4. BUN
5. TC
6. UA
7. HbA1c


## Models

Six machine-learning/statistical models were evaluated:

- Logistic Regression (LR)
- LASSO
- Random Forest (RF)
- XGBoost
- LightGBM
- CatBoost

The final selected model was CatBoost.

The locked CatBoost model used for the manuscript analyses is provided in two formats:

```text
model/locked/model_cat.rds
model/locked/model_cat.cbm
```

The `.rds` object is the R representation used in the analysis workflow. The `.cbm` file is the CatBoost native model format.

Prediction equivalence between the two formats was verified using the synthetic example data. The maximum absolute difference in predicted probabilities was 0 in that verification.


## Important distinction: locked models and model retraining

The public repository distributes the final locked CatBoost model used for the reported manuscript analyses and external validation, in both RDS and native CatBoost formats.

The training scripts are provided to document the development and tuning workflow for all six candidate models. The manuscript-level all-model comparison was locally verified using the corresponding locked manuscript model objects. Non-final locked candidate-model objects are not distributed in the public repository.

Exact retraining of stochastic machine-learning algorithms may depend on software versions, implementation details, and the training environment. In particular, rerunning the current cleaned XGBoost training workflow may produce a boosting-round selection that differs from the locked manuscript XGBoost model.

Accordingly, newly retrained candidate models should not be interpreted as exact replacements for the locked objects used to obtain the reported manuscript-level all-model comparison. The supplied locked CatBoost model is the reference object for exact reuse of the reported final model.


## Main analysis workflow

The principal analysis scripts are located in:

```text
RR/main/
```

They are organized in the following order:

### `00_setup.R`

Defines the random seed, repository paths, predictor names, and common analysis settings.

### `01_prepare_charls_data.R`

Extracts and combines the required CHARLS source variables, constructs treatment indicators, and applies the initial eligibility logic required for MetS ascertainment.

### `02_define_mets_and_complete_cases.R`

Constructs the MetS outcome, applies age and prespecified plausibility criteria, performs required unit conversions, and creates the complete-case analytic dataset.

### `03_descriptive_analysis_and_split.R`

Produces descriptive summaries and creates the stratified 70:30 training/testing split using random seed 123.

### `04_feature_selection.R`

Performs LASSO predictor selection and collinearity assessment.

### `05_train_models.R`

Implements training and tuning of the six candidate models.

### `06_evaluate_models.R`

Evaluates candidate-model objects on the testing dataset and generates model-performance, discrimination, calibration, ROC, and decision-curve outputs. The manuscript-level all-model results were locally verified using the corresponding locked manuscript model objects; only the final locked CatBoost model is distributed publicly.

### `07_shap_analysis.R`

Performs SHapley Additive exPlanations (SHAP) analysis for the locked CatBoost model.

### `08_prepare_external_validation.R`

Processes the external-validation data using the predefined study rules.

### `09_external_validation.R`

Evaluates the locked CatBoost model in the external-validation cohort.


## Sensitivity and subgroup analyses

Additional analyses are located in:

```text
RR/sensitivity/
```

### `01_diagnostic_overlap_sensitivity.R`

Examines correlations between retained predictors and excluded diagnostic components and evaluates CatBoost performance after removing BMI, HbA1c, or both.

### `02_lr_standardization_sensitivity.R`

Compares logistic-regression results with and without predictor standardization.

### `03_threshold_sensitivity.R`

Evaluates classification performance across several probability thresholds in the testing and external-validation datasets.

### `04_probability_stratification_sensitivity.R`

Evaluates alternative exploratory probability-stratification cutoffs.

### `05_shap_zero_crossing_bootstrap.R`

Estimates model-based SHAP zero-crossing reference points and their bootstrap confidence intervals using the fixed locked CatBoost model.

These zero-crossing values are descriptive, model-specific reference points and should not be interpreted as clinical diagnostic thresholds.

### `06_sex_and_age_subgroup_analysis.R`

Evaluates a sensitivity model including sex and reports performance of the original locked CatBoost model across sex and age subgroups.


## Reproducing the analysis

### 1. Software environment

The analyses were performed using R 4.5.1.

Package versions used in the verified local environment are recorded in:

```text
environment/package_versions.csv
environment/sessionInfo.txt
```

The main packages include:

- catboost 1.2.7
- caret 7.0.1
- dplyr 1.1.4
- glmnet 4.1.10
- lightgbm 4.6.0
- pROC 1.19.0.1
- PRROC 1.4
- randomForest 4.7.1.2
- shapviz 0.10.3
- shiny 1.13.0
- xgboost 3.2.0.1


### 2. Data placement

Raw data are not distributed with this repository.

For local reproduction, place the required CHARLS 2015 source files under:

```text
data/raw/CHARLS2015/
```

The analysis expects this location through:

```r
charls_raw_dir <- file.path("data", "raw", "CHARLS2015")
```

External-validation source data, when available to authorized investigators, should be placed under:

```text
data/raw/external_validation/
```

The analysis expects:

```r
external_raw_dir <- file.path("data", "raw", "external_validation")
```

The repository should be used with its root directory as the working directory.


### 3. Main analysis order

After the required source data have been placed in the expected locations, run:

```text
RR/main/00_setup.R
RR/main/01_prepare_charls_data.R
RR/main/02_define_mets_and_complete_cases.R
RR/main/03_descriptive_analysis_and_split.R
RR/main/04_feature_selection.R
RR/main/05_train_models.R
RR/main/06_evaluate_models.R
RR/main/07_shap_analysis.R
```
Scripts 00–05 document and reproduce the data-processing, feature-selection, and candidate-model training workflow for authorized users with access to the required source data. The manuscript-level all-model evaluation in script 06 was locally verified using the corresponding locked manuscript model objects. In a public clone, exact reproduction of the manuscript-level all-model comparison requires the corresponding local candidate-model objects or retraining. Exact reuse of the reported final model is supported by the supplied locked CatBoost RDS and CBM files.
External-validation scripts require the non-public external-validation data:

```text
RR/main/08_prepare_external_validation.R
RR/main/09_external_validation.R
```

The sensitivity scripts can then be run as applicable after the required processed datasets and locked model are available.


## Synthetic prediction example

No real participant data are required to test the supplied locked CatBoost model.

A synthetic example dataset is provided at:

```text
example/example_input.csv
```

It contains the seven required predictors:

```text
age
BMI
pulse
BUN
TC
UA
HbA1c
```

The synthetic data can be regenerated using:

```r
source("example/00_create_example_input.R")
```

Predictions from the locked R model can be generated using:

```r
source("example/predict_example.R")
```

The example script validates the required predictor names and order and converts the prediction matrix explicitly to double precision before creating the CatBoost prediction pool.


## Native CatBoost model verification

The equivalence between the RDS CatBoost model and the native CatBoost model can be checked using:

```r
source("example/01_verify_native_catboost_model.R")
```

In the verified repository version, the two model formats produced identical probabilities for the supplied synthetic examples.


## Shiny application

The source code for the interactive MetS identification tool is provided in:

```text
app/app.R
```

From the repository root, the application can be started locally with:

```r
shiny::runApp("app")
```

The application uses the locked CatBoost model and requires the following seven inputs:

- age
- BMI
- pulse
- BUN
- TC
- UA
- HbA1c

The accepted input ranges in the application correspond to the observed ranges in the model-development training dataset. These ranges are intended to reduce extrapolation and are distinct from the prespecified plausibility limits used during data cleaning.

The application reports the model-estimated probability of prevalent MetS and an exploratory probability category.

It should not be interpreted as a tool for predicting future MetS incidence or as a substitute for clinical diagnosis.


## Probability thresholds and exploratory stratification

Different probability cutoffs in this repository serve different purposes.

The CatBoost training-derived Youden threshold was approximately:

```text
0.2498
```

This threshold was used to describe classification performance at a data-derived operating point.

Separately, exploratory probability categories were defined as:

- Low: < 0.15
- Intermediate: 0.15–0.33
- High: > 0.33

The 0.15 and 0.33 cutoffs were used for exploratory probability stratification and should not be interpreted as validated clinical diagnostic or treatment thresholds.

Sensitivity analyses for alternative thresholds and probability-stratification cutoffs are provided in:

```text
RR/sensitivity/03_threshold_sensitivity.R
RR/sensitivity/04_probability_stratification_sensitivity.R
```


## Data availability and privacy

The CHARLS source data are not redistributed in this repository. Researchers who wish to reproduce analyses using CHARLS should obtain access to the source data through the applicable CHARLS data-access procedures.

The external-validation dataset is not publicly shared because it contains individual-level clinical data and is subject to institutional and privacy restrictions.

Participant-level processed datasets, participant-level model predictions, and participant-level SHAP values are also excluded from the public repository.

The repository instead provides analysis code, aggregate results, a locked final CatBoost model, synthetic example inputs, and the source code for the Shiny application.


## Interpretation and intended use

This model was developed for identification of **prevalent MetS** in a cross-sectional setting.

The model output should therefore be interpreted as an estimated probability of current MetS classification based on the seven model predictors, rather than as a longitudinal estimate of future disease risk.

The model and application are provided for research and educational purposes. They are not intended to replace formal clinical assessment or diagnostic criteria.


## Reproducibility notes

The public repository is designed to support:

1. inspection of the complete analysis workflow;
2. reproduction of analyses by authorized users with access to the required source datasets;
3. reuse of the locked final CatBoost model without access to participant-level training data;
4. verification of model prediction using synthetic inputs;
5. inspection and local execution of the Shiny application source code.

Participant-level datasets are intentionally excluded to protect data privacy and comply with data-access restrictions.


## Citation

Citation information for the associated manuscript and archived repository will be added after final publication and repository archiving.


