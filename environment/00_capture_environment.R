# ==============================================================================
# 00_capture_environment.R
#
# Capture the software environment used for the reproducibility package.
#
# Outputs:
#   environment/package_versions.csv
#   environment/sessionInfo.txt
#
# IMPORTANT:
# - No participant-level data are read.
# - No model is trained or modified.
# - No application source file is modified.
# ==============================================================================


# ==============================================================================
# 1. Output directory
# ==============================================================================

environment_dir <- "environment"

dir.create(
  environment_dir,
  showWarnings = FALSE,
  recursive = TRUE
)


# ==============================================================================
# 2. Packages required by the analysis and Shiny application
# ==============================================================================

required_packages <- c(
  "catboost",
  "caret",
  "dplyr",
  "glmnet",
  "lightgbm",
  "pROC",
  "PRROC",
  "randomForest",
  "shapviz",
  "shiny",
  "xgboost"
)


# ==============================================================================
# 3. Check package availability
# ==============================================================================

package_available <- vapply(
  required_packages,
  requireNamespace,
  quietly = TRUE,
  FUN.VALUE = logical(1)
)


if (!all(package_available)) {
  
  missing_packages <- required_packages[
    !package_available
  ]
  
  stop(
    paste0(
      "Required package(s) not installed: ",
      paste(
        missing_packages,
        collapse = ", "
      )
    )
  )
}


# ==============================================================================
# 4. Capture package versions
# ==============================================================================

package_versions <- data.frame(
  
  Package = required_packages,
  
  Version = vapply(
    required_packages,
    function(pkg) {
      as.character(
        packageVersion(pkg)
      )
    },
    FUN.VALUE = character(1)
  ),
  
  stringsAsFactors = FALSE
)


package_versions_file <- file.path(
  environment_dir,
  "package_versions.csv"
)


write.csv(
  package_versions,
  package_versions_file,
  row.names = FALSE
)


# ==============================================================================
# 5. Capture R session information
# ==============================================================================

session_info <- capture.output(
  sessionInfo()
)


session_info_file <- file.path(
  environment_dir,
  "sessionInfo.txt"
)


writeLines(
  session_info,
  con = session_info_file
)


# ==============================================================================
# 6. Basic privacy / local-path scan
#
# This is only a lightweight predefined-pattern check.
# A broader repository-wide privacy/path audit will be performed separately
# before public release.
# ==============================================================================

files_to_scan <- c(
  package_versions_file,
  session_info_file
)


scan_text <- paste(
  unlist(
    lapply(
      files_to_scan,
      function(f) {
        readLines(
          f,
          warn = FALSE,
          encoding = "UTF-8"
        )
      }
    )
  ),
  collapse = "\n"
)


sensitive_patterns <- c(
  "D:/R/",
  "D:\\\\R\\\\",
  "R指路",
  "2015年charls总数据",
  "外部数据原始",
  "外部原始数据",
  "shinyapps.io",
  "rsconnect",
  "token",
  "secret",
  "password"
)


pattern_detected <- vapply(
  sensitive_patterns,
  function(pattern) {
    grepl(
      tolower(pattern),
      tolower(scan_text),
      fixed = TRUE
    )
  },
  FUN.VALUE = logical(1)
)


privacy_check_passed <- !any(
  pattern_detected
)


# ==============================================================================
# 7. Console report
# ==============================================================================

cat("\n")
cat("============================================================\n")
cat("Reproducibility Environment Capture\n")
cat("============================================================\n")


cat(
  "\nR version:\n",
  R.version.string,
  "\n",
  sep = ""
)


cat(
  "\nRecorded package versions:\n\n"
)

print(
  package_versions,
  row.names = FALSE
)


cat(
  "\nOutput files:\n"
)

cat(
  "  ",
  package_versions_file,
  "\n",
  sep = ""
)

cat(
  "  ",
  session_info_file,
  "\n",
  sep = ""
)


cat(
  "\nPrivacy/path check:\n"
)

if (privacy_check_passed) {
  
  cat(
    "No predefined local path/account patterns detected.\n"
  )
  
} else {
  
  cat(
    "WARNING: One or more predefined patterns were detected:\n"
  )
  
  print(
    sensitive_patterns[
      pattern_detected
    ]
  )
}


cat("\nReproducibility checks\n")
cat("----------------------\n")

cat(
  "All required packages installed: ",
  all(package_available),
  "\n",
  sep = ""
)

cat(
  "Shiny dependency recorded: ",
  "shiny" %in% package_versions$Package,
  "\n",
  sep = ""
)

cat(
  "Package versions saved: ",
  file.exists(package_versions_file),
  "\n",
  sep = ""
)

cat(
  "Session information saved: ",
  file.exists(session_info_file),
  "\n",
  sep = ""
)

cat(
  "Participant-level data read: No\n"
)

cat(
  "Model trained or modified: No\n"
)

cat(
  "Locked model modified: No\n"
)

cat(
  "Shiny application modified: No\n"
)

cat(
  "Basic privacy/path check passed: ",
  privacy_check_passed,
  "\n",
  sep = ""
)

cat("============================================================\n")