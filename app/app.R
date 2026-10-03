# ==============================================================================
# app.R
#
# Shiny application for the locked final CatBoost model for identification
# of prevalent metabolic syndrome (MetS).
#
# Final predictors:
#   age, BMI, pulse, BUN, TC, UA, HbA1c
#
# IMPORTANT:
# - The app uses the locked final CatBoost model.
# - The model is NOT retrained by this application.
# - The model identifies/classifies prevalent MetS and does not predict
#   future MetS incidence.
# - Inputs are restricted to the observed ranges of the model-development
#   training data to reduce extrapolation.
# - The probability strata shown below are exploratory and are not
#   validated clinical decision thresholds.
# ==============================================================================


library(shiny)
library(catboost)


# ==============================================================================
# 1. Required predictors
# ==============================================================================

required_predictors <- c(
  "age",
  "BMI",
  "pulse",
  "BUN",
  "TC",
  "UA",
  "HbA1c"
)


# ==============================================================================
# 2. Locate locked CatBoost model
#
# Supported repository layouts:
#
# A. Repository root as working directory:
#      model/locked/model_cat.rds
#
# B. app/ as working directory:
#      ../model/locked/model_cat.rds
#
# No absolute local path is used.
# ==============================================================================

model_candidates <- c(
  file.path(
    "model",
    "locked",
    "model_cat.rds"
  ),
  file.path(
    "..",
    "model",
    "locked",
    "model_cat.rds"
  )
)


existing_models <- model_candidates[
  file.exists(model_candidates)
]


if (length(existing_models) == 0) {
  
  stop(
    paste0(
      "Locked CatBoost model not found. ",
      "Expected model/locked/model_cat.rds ",
      "relative to the repository structure."
    )
  )
}


model_file <- existing_models[1]


cat_model <- readRDS(
  model_file
)


# ==============================================================================
# 3. Observed training-data ranges
#
# These values were calculated from the final model-development training set
# (n = 6,907).
#
# They are observed training-data ranges, NOT prespecified data-cleaning
# plausibility limits.
# ==============================================================================

input_ranges <- list(
  
  age = c(
    min = 45,
    max = 93
  ),
  
  BMI = c(
    min = 13.426286,
    max = 58.99604
  ),
  
  pulse = c(
    min = 35.666667,
    max = 133.66667
  ),
  
  BUN = c(
    min = 5.042016,
    max = 51.54061
  ),
  
  TC = c(
    min = 69.884171,
    max = 616.60229
  ),
  
  UA = c(
    min = 1.0,
    max = 14.4
  ),
  
  HbA1c = c(
    min = 3.8,
    max = 17.1
  )
)


# ==============================================================================
# 4. Exploratory probability strata
#
#   Low          < 0.15
#   Intermediate 0.15–0.33
#   High          > 0.33
#
# These strata are exploratory and are NOT clinical diagnostic or treatment
# thresholds.
# ==============================================================================

low_cutoff <- 0.15
high_cutoff <- 0.33


# ==============================================================================
# 5. User interface
# ==============================================================================

ui <- fluidPage(
  
  tags$head(
    
    tags$style(
      HTML(
        "
        .result-box {
          color: white;
          padding: 35px;
          border-radius: 18px;
          text-align: center;
          margin-top: 10px;
          box-shadow: 2px 2px 12px rgba(0,0,0,0.10);
        }

        .probability-value {
          font-size: 56px;
          font-weight: bold;
          margin-bottom: 0;
        }

        .probability-label {
          margin-top: 5px;
        }

        .app-note {
          font-size: 12px;
          color: #666666;
          margin-top: 35px;
          line-height: 1.6;
          border-top: 1px solid #eeeeee;
          padding-top: 18px;
        }

        .range-note {
          font-size: 12px;
          color: #666666;
          margin-top: 12px;
          line-height: 1.5;
        }
        "
      )
    )
  ),
  
  
  titlePanel(
    "Metabolic Syndrome (MetS) Identification Tool"
  ),
  
  
  sidebarLayout(
    
    sidebarPanel(
      
      h4(
        "Input Parameters"
      ),
      
      p(
        paste(
          "Enter the seven predictor values to obtain the",
          "model-estimated probability of prevalent MetS."
        )
      ),
      
      
      numericInput(
        "age",
        "Age (years):",
        value = 60,
        min = input_ranges$age["min"],
        max = input_ranges$age["max"]
      ),
      
      
      numericInput(
        "BMI",
        "BMI (kg/m²):",
        value = 24,
        min = input_ranges$BMI["min"],
        max = input_ranges$BMI["max"]
      ),
      
      
      numericInput(
        "pulse",
        "Pulse (beats/min):",
        value = 74,
        min = input_ranges$pulse["min"],
        max = input_ranges$pulse["max"]
      ),
      
      
      numericInput(
        "BUN",
        "BUN (mg/dL):",
        value = 14.8,
        min = input_ranges$BUN["min"],
        max = input_ranges$BUN["max"]
      ),
      
      
      numericInput(
        "TC",
        "Total cholesterol (TC, mg/dL):",
        value = 184,
        min = input_ranges$TC["min"],
        max = input_ranges$TC["max"]
      ),
      
      
      numericInput(
        "UA",
        "Uric acid (UA, mg/dL):",
        value = 4.8,
        min = input_ranges$UA["min"],
        max = input_ranges$UA["max"]
      ),
      
      
      numericInput(
        "HbA1c",
        "HbA1c (%):",
        value = 5.8,
        min = input_ranges$HbA1c["min"],
        max = input_ranges$HbA1c["max"]
      ),
      
      
      tags$div(
        class = "range-note",
        paste(
          "Inputs are restricted to the observed ranges of the",
          "model-development training data to reduce extrapolation",
          "beyond the data used for model development."
        )
      ),
      
      
      br(),
      
      
      actionButton(
        "predict",
        "Calculate Probability",
        class = "btn-primary",
        style = "width: 100%;"
      )
    ),
    
    
    mainPanel(
      
      h3(
        "Model Output"
      ),
      
      
      uiOutput(
        "result_box"
      ),
      
      
      hr(),
      
      
      h4(
        "Exploratory Probability Strata"
      ),
      
      
      tags$ul(
        
        tags$li(
          strong(
            "Low probability: "
          ),
          "<15%"
        ),
        
        tags$li(
          strong(
            "Intermediate probability: "
          ),
          "15%–33%"
        ),
        
        tags$li(
          strong(
            "High probability: "
          ),
          ">33%"
        )
      ),
      
      
      p(
        tags$em(
          paste(
            "These probability strata are exploratory and should not",
            "be interpreted as validated clinical diagnostic or",
            "treatment thresholds."
          )
        )
      ),
      
      
      tags$div(
        
        class = "app-note",
        
        strong(
          "Important note: "
        ),
        
        paste(
          "This tool estimates the probability of prevalent metabolic",
          "syndrome using a cross-sectional classification model.",
          "It does not predict future development of metabolic syndrome."
        ),
        
        br(),
        br(),
        
        paste(
          "The tool is intended for research and educational use and",
          "does not replace clinical assessment or professional medical advice."
        )
      )
    )
  )
)


# ==============================================================================
# 6. Server
# ==============================================================================

server <- function(
    input,
    output,
    session
) {
  
  
  observeEvent(
    input$predict,
    {
      
      
      # ------------------------------------------------------------------------
      # 6.1 Construct one-row prediction input
      # ------------------------------------------------------------------------
      
      new_data <- data.frame(
        
        age =
          as.numeric(
            input$age
          ),
        
        BMI =
          as.numeric(
            input$BMI
          ),
        
        pulse =
          as.numeric(
            input$pulse
          ),
        
        BUN =
          as.numeric(
            input$BUN
          ),
        
        TC =
          as.numeric(
            input$TC
          ),
        
        UA =
          as.numeric(
            input$UA
          ),
        
        HbA1c =
          as.numeric(
            input$HbA1c
          ),
        
        check.names = FALSE
      )
      
      
      # ------------------------------------------------------------------------
      # 6.2 Verify predictor names and order
      # ------------------------------------------------------------------------
      
      validate(
        
        need(
          identical(
            names(new_data),
            required_predictors
          ),
          "Internal predictor specification error."
        )
      )
      
      
      # ------------------------------------------------------------------------
      # 6.3 Verify finite and non-missing inputs
      # ------------------------------------------------------------------------
      
      validate(
        
        need(
          !anyNA(
            new_data
          ),
          "All seven predictor values are required."
        ),
        
        need(
          all(
            vapply(
              new_data,
              function(x) {
                all(
                  is.finite(x)
                )
              },
              logical(1)
            )
          ),
          "All predictor values must be finite numeric values."
        )
      )
      
      
      # ------------------------------------------------------------------------
      # 6.4 Verify inputs are within observed training-data ranges
      # ------------------------------------------------------------------------
      
      validate(
        
        need(
          new_data$age >= input_ranges$age["min"] &&
            new_data$age <= input_ranges$age["max"],
          "Age is outside the observed training-data range."
        ),
        
        need(
          new_data$BMI >= input_ranges$BMI["min"] &&
            new_data$BMI <= input_ranges$BMI["max"],
          "BMI is outside the observed training-data range."
        ),
        
        need(
          new_data$pulse >= input_ranges$pulse["min"] &&
            new_data$pulse <= input_ranges$pulse["max"],
          "Pulse is outside the observed training-data range."
        ),
        
        need(
          new_data$BUN >= input_ranges$BUN["min"] &&
            new_data$BUN <= input_ranges$BUN["max"],
          "BUN is outside the observed training-data range."
        ),
        
        need(
          new_data$TC >= input_ranges$TC["min"] &&
            new_data$TC <= input_ranges$TC["max"],
          "TC is outside the observed training-data range."
        ),
        
        need(
          new_data$UA >= input_ranges$UA["min"] &&
            new_data$UA <= input_ranges$UA["max"],
          "UA is outside the observed training-data range."
        ),
        
        need(
          new_data$HbA1c >= input_ranges$HbA1c["min"] &&
            new_data$HbA1c <= input_ranges$HbA1c["max"],
          "HbA1c is outside the observed training-data range."
        )
      )
      
      
      # ------------------------------------------------------------------------
      # 6.5 Explicitly convert predictor matrix to double precision
      # ------------------------------------------------------------------------
      
      prediction_matrix <- as.matrix(
        new_data[
          ,
          required_predictors,
          drop = FALSE
        ]
      )
      
      
      storage.mode(
        prediction_matrix
      ) <- "double"
      
      
      # ------------------------------------------------------------------------
      # 6.6 CatBoost prediction
      # ------------------------------------------------------------------------
      
      prediction_pool <- catboost.load_pool(
        data = prediction_matrix
      )
      
      
      probability <- as.numeric(
        
        catboost.predict(
          cat_model,
          prediction_pool,
          prediction_type = "Probability"
        )
      )
      
      
      validate(
        
        need(
          length(probability) == 1 &&
            is.finite(probability) &&
            probability >= 0 &&
            probability <= 1,
          "The model did not return a valid probability."
        )
      )
      
      
      # ------------------------------------------------------------------------
      # 6.7 Exploratory probability stratum
      # ------------------------------------------------------------------------
      
      stratum <- if (
        probability < low_cutoff
      ) {
        
        list(
          background = "#2E7D32",
          label = "LOW PROBABILITY"
        )
        
      } else if (
        probability <= high_cutoff
      ) {
        
        list(
          background = "#EF6C00",
          label = "INTERMEDIATE PROBABILITY"
        )
        
      } else {
        
        list(
          background = "#C62828",
          label = "HIGH PROBABILITY"
        )
      }
      
      
      # ------------------------------------------------------------------------
      # 6.8 Render output
      # ------------------------------------------------------------------------
      
      output$result_box <- renderUI(
        
        div(
          
          class = "result-box",
          
          style = paste0(
            "background-color:",
            stratum$background,
            ";"
          ),
          
          div(
            class = "probability-value",
            paste0(
              round(
                probability * 100,
                1
              ),
              "%"
            )
          ),
          
          h3(
            class = "probability-label",
            stratum$label
          ),
          
          p(
            "Model-estimated probability of prevalent MetS"
          )
        )
      )
    }
  )
}


# ==============================================================================
# 7. Run application
# ==============================================================================

shinyApp(
  ui = ui,
  server = server
)