#!/usr/bin/env Rscript
# install_yolor.R
# ============================================================
# One-shot installation script for yolor and its dependencies.
# Run from the terminal:
#   Rscript install_yolor.R
# Or from inside R:
#   source("install_yolor.R")
# ============================================================

message("=== yolor installer ===\n")

# 1. R package dependencies
cran_pkgs <- c(
  "DBI", "RSQLite", "reticulate", "magick",
  "ggplot2", "dplyr", "jsonlite", "yaml",
  "fs", "cli", "rlang", "glue", "tibble",
  # Suggested
  "readr", "purrr", "tidyr", "scales",
  "devtools", "testthat", "knitr", "rmarkdown"
)

to_install <- cran_pkgs[!sapply(cran_pkgs, requireNamespace, quietly = TRUE)]

if (length(to_install) > 0) {
  message("Installing CRAN packages: ", paste(to_install, collapse = ", "))
  install.packages(to_install,
                   repos = "https://cloud.r-project.org",
                   quiet = FALSE)
} else {
  message("All CRAN dependencies already installed.")
}

# 2. Install yolor from GitHub
message("\nInstalling yolor from GitHub...")
devtools::install_github("Lalitgis/yolor", upgrade = "never")

# 3. Set up the Python / Ultralytics backend
message("\nSetting up Python backend (Ultralytics YOLOv8)...")
library(yolor)
yolo_setup()   # creates 'yolor' virtualenv with ultralytics + torch

message("\n=== Installation complete ===")
message("Add this line to your script to activate the Python environment:")
message('  reticulate::use_virtualenv("yolor")')
message("\nQuick test:")
message('  library(yolor)')
message('  db <- yolor_example_db()')
message('  ds <- sl_read_db(db)')
message('  print(ds)')
