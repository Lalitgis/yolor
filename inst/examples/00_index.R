# =============================================================
#  yolor — Example Scripts Index
#
#  Run this script to see available examples and open them.
#  All examples are installed with the package and work
#  without modifying any file paths for the demo sections.
# =============================================================

library(yolor)

# ── List all examples ─────────────────────────────────────────

examples_dir <- system.file("examples", package = "yolor")
example_files <- list.files(examples_dir, pattern = "\\.R$",
                             full.names = FALSE)

cat("=== yolor Example Scripts ===\n\n")
descriptions <- c(
  "01_object_detection.R" = "Object Detection  — annotate → train → detect → export",
  "02_segmentation.R"     = "Segmentation      — train YOLOv8-seg → masks → metrics",
  "03_metrics.R"          = "Accuracy Metrics  — mAP, PR curve, F1, confusion matrix"
)
for (f in example_files) {
  cat(sprintf("  %s\n    %s\n\n", f,
              descriptions[[f]]))
}

# ── Open an example in RStudio ────────────────────────────────

open_example <- function(filename) {
  path <- system.file("examples", filename, package = "yolor")
  if (!nzchar(path)) stop("Example not found: ", filename)

  # RStudio
  if (requireNamespace("rstudioapi", quietly = TRUE) &&
      rstudioapi::isAvailable()) {
    rstudioapi::navigateToFile(path)
    message("Opened in RStudio: ", path)
  } else {
    # fallback — copy to working directory and open
    dest <- file.path(getwd(), filename)
    file.copy(path, dest, overwrite = TRUE)
    message("Copied to: ", dest)
    if (.Platform$OS.type == "windows") shell.exec(dest)
    else if (Sys.info()["sysname"] == "Darwin") system(paste("open", dest))
    else system(paste("xdg-open", dest))
  }
  invisible(path)
}

# ── Usage ─────────────────────────────────────────────────────

cat("To open an example in RStudio:\n")
cat('  open_example("01_object_detection.R")\n')
cat('  open_example("02_segmentation.R")\n')
cat('  open_example("03_metrics.R")\n\n')

cat("To get the full path to any example:\n")
cat('  system.file("examples", "01_object_detection.R",\n')
cat('              package = "yolor")\n\n')

cat("To run the metrics example directly (no Python needed):\n")
cat('  source(system.file("examples", "03_metrics.R",\n')
cat('                     package = "yolor"))\n')
