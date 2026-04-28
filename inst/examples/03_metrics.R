# =============================================================
#  Example 03: Accuracy Metrics — Calculation & Visualisation
#  yolor package
#
#  Covers:
#   1. Metrics from a trained model (Ultralytics validation)
#   2. Metrics from raw prediction tibbles (pure R, no Python)
#   3. All 6 visualisation types
#   4. Exporting metrics (CSV, JSON, PNG, HTML report)
#   5. Comparing two models side-by-side
# =============================================================

library(yolor)
library(tibble)   # for tibble()
library(dplyr)    # for filter(), mutate()


# ══════════════════════════════════════════════════════════════
#  PART A — Metrics from a trained model (requires Python)
# ══════════════════════════════════════════════════════════════

# ── A1: Run after training ────────────────────────────────────

# After running Example 01 or 02:
# model   <- yolo_model("runs/detect/detect_example/weights/best.pt")
# metrics <- yolo_metrics(model, data = "path/to/data.yaml")

# Full validation — computes:
#   mAP@0.5, mAP@0.5:0.95, Precision, Recall, F1
#   per-class AP, PR curve, F1-confidence curve, confusion matrix

# ── A2: Quick benchmark ───────────────────────────────────────

# bench <- yolo_benchmark(model, data = "path/to/data.yaml")
# print(bench)
# plot(bench)   # per-class bar chart


# ══════════════════════════════════════════════════════════════
#  PART B — Pure-R metrics (no Python needed)
#  Works with any prediction + ground-truth tibbles
# ══════════════════════════════════════════════════════════════

# ── B1: Build ground truth and predictions ───────────────────

# Ground truth: what was actually in the images
ground_truth <- tibble(
  image      = c("img01.jpg","img01.jpg","img01.jpg",
                 "img02.jpg","img02.jpg",
                 "img03.jpg","img03.jpg","img03.jpg",
                 "img04.jpg","img04.jpg"),
  class_name = c("cat","dog","cat",
                 "cat","bird",
                 "dog","dog","cat",
                 "bird","cat"),
  xmin = c( 10, 200,  50, 100,  30,  20, 300,  80, 150,  60),
  ymin = c( 10, 150,  80,  40,  60,  30, 200,  90,  20, 100),
  xmax = c(110, 320, 150, 220, 130, 120, 420, 180, 250, 160),
  ymax = c(110, 270, 180, 160, 160, 130, 320, 190, 120, 200)
)

# Predictions: what the model detected
predictions <- tibble(
  image      = c("img01.jpg","img01.jpg","img01.jpg",
                 "img02.jpg","img02.jpg","img02.jpg",
                 "img03.jpg","img03.jpg","img03.jpg",
                 "img04.jpg","img04.jpg"),
  class_name = c("cat","dog","cat",
                 "cat","bird","cat",   # extra false positive cat
                 "dog","dog","cat",
                 "bird","cat"),
  confidence = c(0.95, 0.88, 0.76,
                 0.91, 0.83, 0.32,    # low-conf false positive
                 0.97, 0.71, 0.85,
                 0.89, 0.62),
  xmin = c( 12, 202,  52, 103,  32,  400,  22, 302,  82, 152,  62),
  ymin = c( 12, 152,  82,  42,  62,  200,  32, 202,  92,  22, 102),
  xmax = c(112, 322, 152, 222, 132,  500, 122, 422, 182, 252, 162),
  ymax = c(112, 272, 182, 162, 162,  300, 132, 322, 192, 122, 202)
)

cat("Ground truth boxes:", nrow(ground_truth), "\n")
cat("Prediction boxes  :", nrow(predictions),  "\n")
cat("Classes           :", paste(unique(ground_truth$class_name),
                                  collapse = ", "), "\n\n")


# ── B2: Compute metrics ───────────────────────────────────────

metrics <- metrics_from_predictions(
  predictions  = predictions,
  ground_truth = ground_truth,
  iou_thresh   = 0.5    # a detection counts as TP if IoU >= 0.5
)

# Overall metrics
print(metrics)


# ── B3: Inspect each component ───────────────────────────────

# Overall summary table
cat("\n── Overall Metrics ──\n")
print(metrics$overall)

# Per-class breakdown
cat("\n── Per-Class Metrics ──\n")
print(metrics$per_class)

# TP / FP / FN counts
cat("\n── Detection Counts ──\n")
metrics$per_class[, c("class_name","TP","FP","FN","n_gt","n_pred")]

# PR curve data (recall + precision at each confidence threshold)
cat("\n── PR Curve (first 6 rows) ──\n")
head(metrics$pr_curve)

# F1 curve data (F1 score at each confidence threshold)
cat("\n── F1 Curve (first 6 rows) ──\n")
head(metrics$f1_curve)

# Confusion matrix (rows = predicted, cols = actual)
cat("\n── Confusion Matrix ──\n")
print(metrics$conf_matrix)


# ══════════════════════════════════════════════════════════════
#  PART C — Visualisation (6 plot types)
# ══════════════════════════════════════════════════════════════

# ── C1: Precision-Recall Curve ───────────────────────────────
# Shows the trade-off between precision and recall at different
# confidence thresholds. Higher area under curve = better model.
pr_plot <- plot(metrics, type = "pr")
print(pr_plot)


# ── C2: F1-Confidence Curve ──────────────────────────────────
# Shows F1 score at each confidence threshold.
# The dashed line marks the optimal confidence threshold.
f1_plot <- plot(metrics, type = "f1")
print(f1_plot)


# ── C3: Confusion Matrix Heatmap ─────────────────────────────
# Normalised by ground-truth column totals (recall-normalised).
# Diagonal = correctly predicted. Off-diagonal = errors.
cm_plot <- plot(metrics, type = "confusion")
print(cm_plot)


# ── C4: Per-Class Bar Chart ──────────────────────────────────
# Grouped bars showing Precision / Recall / F1 / AP per class.
# Useful for identifying which classes the model struggles with.
bar_plot <- plot(metrics, type = "bar")
print(bar_plot)


# ── C5: Metric Radar Chart ───────────────────────────────────
# Spider chart showing all 5 overall metrics at a glance.
radar_plot <- plot(metrics, type = "radar")
print(radar_plot)


# ── C6: Dashboard (all plots in one figure) ──────────────────
# Requires the 'patchwork' package:
# install.packages("patchwork")
dashboard <- plot(metrics, type = "dashboard")
print(dashboard)


# ── C7: All plots as a named list ────────────────────────────
all_plots <- plot(metrics, type = "all")
names(all_plots)   # "pr", "f1", "confusion", "bar", "radar"

# Access individually
print(all_plots$pr)
print(all_plots$confusion)


# ── C8: Filter to specific classes ───────────────────────────
plot(metrics, type = "bar", classes = c("cat", "dog"))
plot(metrics, type = "pr",  classes = c("cat", "dog"))


# ══════════════════════════════════════════════════════════════
#  PART D — Export everything
# ══════════════════════════════════════════════════════════════

output_dir <- file.path(tempdir(), "metrics_output")

exported <- metrics_export(
  metrics,
  dir    = output_dir,
  plots  = c("pr","f1","confusion","bar","radar"),
  width  = 8,
  height = 6,
  dpi    = 150,
  html   = TRUE,   # generates metrics_report.html
  pdf    = FALSE,  # set TRUE if rmarkdown is installed
  prefix = "example"
)

# List all exported files
cat("\n── Exported Files ──\n")
for (nm in names(exported)) {
  cat(sprintf("  %-20s : %s\n", nm, basename(exported[[nm]])))
}

cat("\nAll outputs saved to:", output_dir, "\n")

# Open the HTML report in your browser
# browseURL(exported$html_report)


# ══════════════════════════════════════════════════════════════
#  PART E — Compare two models
# ══════════════════════════════════════════════════════════════

# Simulate a second model with slightly different predictions
predictions_v2 <- dplyr::mutate(
  predictions,
  confidence = pmin(confidence * 1.1, 1.0),   # slightly higher conf
  xmin = xmin + 2, ymin = ymin + 2,
  xmax = xmax + 2, ymax = ymax + 2
)

metrics_v2 <- metrics_from_predictions(predictions_v2, ground_truth)

# Side-by-side comparison
comparison <- metrics_compare(
  metrics,
  metrics_v2,
  labels = c("YOLOv8n (baseline)", "YOLOv8n (tuned)")
)

# Bar chart comparing both models across all metrics
print(comparison$plot)

# Delta table — which model wins on each metric?
cat("\n── Model Comparison Table ──\n")
print(comparison$table)

# Export comparison plot
ggplot2::ggsave(
  file.path(output_dir, "model_comparison.png"),
  comparison$plot,
  width = 9, height = 5, dpi = 150, bg = "white"
)
