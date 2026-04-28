# =============================================================
#  Example 02: Instance Segmentation — Full Training Workflow
#  yolor package
#
#  Instance segmentation predicts a pixel-level mask for each
#  detected object, not just a bounding box.
#
#  Covers:
#   1. Prepare a segmentation dataset (Roboflow or ShinyLabel)
#   2. Train a YOLOv8 segmentation model
#   3. Run segmentation inference
#   4. Visualise masks and export results
#   5. Calculate segmentation metrics
#
#  Requirements:
#   - Annotations must include polygon masks (not just boxes)
#   - In Roboflow: export as "YOLOv8 Segmentation"
#   - In ShinyLabel: use polygon annotation mode
# =============================================================

library(yolor)

# ── Step 0: Python backend ───────────────────────────────────
# yolo_setup()
# reticulate::use_virtualenv("yolor")


# ── Step 1: Load segmentation dataset ────────────────────────

## Option A — Roboflow YOLOv8 Segmentation export
##   In Roboflow: Export → YOLOv8 Segmentation → Download ZIP → unzip
# yaml_path <- rf_load_yolo("path/to/roboflow_seg_export/")

## Option B — Roboflow COCO JSON export then convert
# yaml_path <- rf_coco_to_yolo(
#   dataset_dir = "path/to/roboflow_coco_export/",
#   output_dir  = "path/to/seg_dataset_yolo/"
# )

## Option C — manual path to an existing segmentation data.yaml
# yaml_path <- "path/to/seg_dataset/data.yaml"

## For demo purposes — use the bundled example with bounding boxes
## (real segmentation requires polygon annotations)
dataset_dir <- file.path(tempdir(), "seg_dataset")
db        <- yolor_example_db()
ds        <- sl_read_db(db)
yaml_path <- sl_export_dataset(ds, dataset_dir,
                                val_split = 0.2, seed = 42,
                                copy_images = TRUE)

cat("Dataset ready:", yaml_path, "\n")
yolo_validate_dataset(dataset_dir)


# ── Step 2: Load segmentation model ──────────────────────────

# YOLOv8 segmentation models are named with "-seg" suffix
# Available: yolov8n-seg, yolov8s-seg, yolov8m-seg,
#            yolov8l-seg, yolov8x-seg

seg_model <- yolo_model(
  weights = "yolov8n-seg",   # nano segmentation model
  task    = "segment",       # <-- key difference from detection
  device  = "cpu"            # or "cuda" / "mps"
)

print(seg_model)


# ── Step 3: Train segmentation model ─────────────────────────

seg_result <- yolo_train(
  seg_model,
  data      = yaml_path,
  epochs    = 50,
  imgsz     = 640,
  batch     = 8,        # segmentation needs more memory than detection
  lr0       = 0.01,
  patience  = 20,
  project   = "runs",
  name      = "seg_example"
)

print(seg_result)
cat("Best segmentation weights:", seg_result$best_weights, "\n")


# ── Step 4: Run segmentation inference ───────────────────────

trained_seg <- yolo_model(
  seg_result$best_weights,
  task = "segment"
)

seg_detections <- yolo_detect(
  trained_seg,
  images = "path/to/new_images/",   # replace with real folder
  conf   = 0.4,
  imgsz  = 640
)

print(seg_detections)

# Flat tibble — contains bounding box coords + mask info
seg_df <- as_tibble(seg_detections)
head(seg_df)


# ── Step 5: Visualise segmentation results ───────────────────

# Plot masks overlaid on image
plot(seg_detections)

# Plot a specific image
# plot(seg_detections, image = "path/to/new_images/img001.jpg")


# ── Step 6: Evaluate segmentation model ──────────────────────

# Quick benchmark
bench <- yolo_benchmark(
  trained_seg,
  data  = yaml_path,
  split = "val"
)
print(bench)
plot(bench)

# Full metrics — mAP, PR curve, confusion matrix
metrics <- yolo_metrics(
  trained_seg,
  data  = yaml_path,
  split = "val"
)
print(metrics)

# Visualise all metric plots
plot(metrics, type = "pr")          # Precision-Recall curve
plot(metrics, type = "f1")          # F1-Confidence curve
plot(metrics, type = "confusion")   # Confusion matrix heatmap
plot(metrics, type = "bar")         # Per-class bar chart
plot(metrics, type = "radar")       # Overall metric spider chart
plot(metrics, type = "dashboard")   # All plots in one panel


# ── Step 7: Export segmentation results ──────────────────────

output_dir <- file.path(tempdir(), "seg_results")
dir.create(output_dir, showWarnings = FALSE)

# Export detections
yolo_export_csv(seg_detections,
                file.path(output_dir, "seg_detections.csv"))

# Export full metrics report
metrics_export(
  metrics,
  dir    = file.path(output_dir, "metrics"),
  html   = TRUE,
  prefix = "seg"
)

cat("All results saved to:", output_dir, "\n")


# ── Tips for better segmentation results ─────────────────────
#
#  1. Use more epochs (100-200) for segmentation vs detection
#  2. Reduce batch size if you get CUDA out-of-memory errors
#  3. imgsz = 640 is standard; use 1280 for small objects
#  4. More training images = better masks (aim for 500+ per class)
#  5. Use yolov8s-seg or yolov8m-seg for better accuracy
#     at the cost of speed
